"""Knowledge Distillation Lightning Modules

This module implements all three multi-teacher KD methods from the AML Final Project:
1. Method 1: CA-WKD (Confidence-Aware Weighted KD)
2. Method 2: α-Guided CA-WKD (with dynamic alpha and gating)
3. Method 3: Adaptive α-Guided KD (most confident teacher selection)
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, KLDivLoss
from torchmetrics.classification import MulticlassAUROC


class CAWeightedKDLitModule(pl.LightningModule):
    """
    Method 1: Confidence-Aware Weighted KD (CA-WKD)
    
    This module implements the CA-WKD method where:
    - Teachers are weighted based on their cross-entropy loss  
    - Total loss is: L_total = L_KL + L_CE (equal weights, no gating)
    - Uses weighted sum of teacher logits for KD
    
    From the paper, formula (1-4):
    L_total = L_KL + L_CE
    where L_KL is the weighted sum of KL divergences
    """
    
    def __init__(
        self,
        teacher_models: List[nn.Module],
        student_model: nn.Module,
        temperature: float = 4.0,
        learning_rate: float = 1e-3,
        num_classes: int = 100
    ):
        """
        Initialize CA-WKD Module (Method 1).
        
        Args:
            teacher_models: List of pre-trained teacher models
            student_model: Student model to be trained
            temperature: Temperature for softening probabilities
            learning_rate: Learning rate for optimizer
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=["teacher_models", "student_model"])
        
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Student model
        if student_model is None:
            raise ValueError("A student model must be provided.")
        self.student = student_model
        
        # Teacher models
        if not teacher_models or len(teacher_models) == 0:
            raise ValueError("Teacher models must be provided for CA-WKD.")
        self.teachers = teacher_models
        self.num_teachers = len(self.teachers)
        
        # Freeze teachers
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
        
        # Loss functions
        self.kl_div_loss = KLDivLoss(reduction='batchmean')
        self.ce_loss = CrossEntropyLoss()
        
        # Metrics
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
    
    def forward(self, x):
        """Forward pass through student model."""
        return self.student(x)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return [optimizer], [scheduler]
    
    def compute_weights(
        self, 
        teacher_logits: List[torch.Tensor], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic weights for each teacher based on their CE loss.
        
        Formula from paper (Eq. 1):
        w_k = (1/(K-1)) * [1 - exp(L_CE^k) / sum_j exp(L_CE^j)]
        
        Args:
            teacher_logits: List of logits from each teacher
            labels: Ground truth labels
            
        Returns:
            Weights for each teacher, shape (batch_size, num_teachers)
        """
        K = len(teacher_logits)
        
        # Compute CE loss for each teacher (per-sample)
        ce_losses = []
        for logit in teacher_logits:
            ce = F.cross_entropy(logit, labels, reduction='none')
            ce_losses.append(ce)
        
        ce_losses = torch.stack(ce_losses, dim=1)  # (batch_size, K)
        
        # Compute exponential weights
        exp_losses = torch.exp(ce_losses)
        sum_exp_losses = torch.sum(exp_losses, dim=1, keepdim=True)
        
        weights = (1.0 / (K - 1)) * (1.0 - (exp_losses / sum_exp_losses))
        
        return weights
    
    def compute_losses(
        self, 
        student_logits: torch.Tensor, 
        labels: torch.Tensor, 
        teacher_logits: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total loss for CA-WKD (Method 1).
        
        L_total = L_KL + L_CE (equal weights, Eq. 4)
        
        Args:
            student_logits: Logits from student model
            labels: Ground truth labels
            teacher_logits: List of logits from teacher models
            
        Returns:
            Total loss value
        """
        device = student_logits.device
        
        # Compute teacher weights (Eq. 1)
        weights = self.compute_weights(teacher_logits, labels)  # (batch_size, K)
        
        # Compute weighted KD loss (Eq. 2-3)
        kd_loss = 0.0
        for k in range(self.num_teachers):
            z_T_k = teacher_logits[k]
            log_student = F.log_softmax(student_logits / self.temperature, dim=1)
            soft_teacher = F.softmax(z_T_k / self.temperature, dim=1)
            loss_kd_k = self.kl_div_loss(log_student, soft_teacher) * (self.temperature ** 2)
            weight_k = torch.mean(weights[:, k])
            kd_loss += weight_k * loss_kd_k
        
        # Compute CE loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Total loss (Eq. 4): equal weights for KD and CE
        loss_total = kd_loss + ce_loss
        
        # Logging
        self.log_dict({
            'train/loss_kl': kd_loss,
            'train/loss_ce': ce_loss,
            'train/loss_total': loss_total,
        }, on_epoch=True, prog_bar=True)
        
        return loss_total
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        teacher_logits = []
        with torch.no_grad():
            for i in range(self.num_teachers):
                teacher_images = batch[f'teacher_input_{i}'].to(self.device)
                logits = self.teachers[i](teacher_images)
                teacher_logits.append(logits)
        
        loss = self.compute_losses(student_logits, labels, teacher_logits)
        return loss
    
    def shared_eval_step(self, batch, stage: str):
        """Shared evaluation step for validation and test."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        teacher_logits = []
        with torch.no_grad():
            for i in range(self.num_teachers):
                teacher_images = batch[f'teacher_input_{i}'].to(self.device)
                logits = self.teachers[i](teacher_images)
                teacher_logits.append(logits)
        
        # Compute loss
        loss = self.compute_losses(student_logits, labels, teacher_logits)
        
        # Metrics
        preds = student_logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # AUROC
        if stage == 'val':
            self.val_auroc(student_logits, labels)
            self.log('val/auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        elif stage == 'test':
            self.test_auroc(student_logits, labels)
            self.log('test/auroc', self.test_auroc, on_epoch=True, prog_bar=True)
        
        self.log(f'{stage}/accuracy', acc, on_epoch=True, prog_bar=True)
        self.log(f'{stage}/loss_total', loss, on_epoch=True, prog_bar=True)
        
        return acc
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.shared_eval_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.shared_eval_step(batch, 'test')


class DynamicKDLitModule(pl.LightningModule):
    """
    Method 2: α-Guided CA-WKD (Dynamic Knowledge Distillation with Gating)
    
    This module implements knowledge distillation with:
    - Dynamic weight computation for each teacher based on their cross-entropy loss
    - Confidence-based gating mechanism to balance soft and hard losses
    - Support for multiple teacher models
    
    From the paper formula (5-6):
    L_total = α * L_KL + (1-α) * L_CE
    where α = sigmoid(γ * (Conf_avg - θ))
    """
    
    def __init__(
        self,
        teacher_models: List[nn.Module],
        student_model: nn.Module,
        temperature: float = 4.0,
        gamma: float = 10.0,
        threshold: float = 0.5,
        learning_rate: float = 1e-3,
        use_soft_loss: bool = True,
        use_hard_loss: bool = True,
        alpha: float = 0.5,
        num_classes: int = 100
    ):
        """
        Initialize Dynamic KD Module (Method 2).
        
        Args:
            teacher_models: List of pre-trained teacher models
            student_model: Student model to be trained
            temperature: Temperature for softening probabilities
            gamma: Scaling factor for sigmoid function in gate computation
            threshold: Threshold for gate activation
            learning_rate: Learning rate for optimizer
            use_soft_loss: Whether to use KL Divergence loss (KD)
            use_hard_loss: Whether to use Cross-Entropy loss
            alpha: Base weighting factor between soft and hard losses
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=["teacher_models", "student_model"])
        
        # Configuration
        self.use_soft_loss = use_soft_loss
        self.use_hard_loss = use_hard_loss
        self.alpha = alpha
        self.temperature = temperature
        self.gamma = gamma
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Validation
        if not (self.use_soft_loss or self.use_hard_loss):
            raise ValueError("At least one of 'use_soft_loss' or 'use_hard_loss' must be True.")
        
        # Student model
        if student_model is None:
            raise ValueError("A student model must be provided.")
        self.student = student_model
        
        # Teacher models
        if self.use_soft_loss:
            if not teacher_models or len(teacher_models) == 0:
                raise ValueError("Teacher models must be provided if 'use_soft_loss' is True.")
            self.teachers = teacher_models
            self.num_teachers = len(self.teachers)
            
            # Freeze teachers
            for teacher in self.teachers:
                teacher.eval()
                for param in teacher.parameters():
                    param.requires_grad = False
                    
            self.kl_div_loss = KLDivLoss(reduction='batchmean')
        else:
            self.teachers = None
            self.num_teachers = 0
        
        # Loss functions
        if self.use_hard_loss:
            self.ce_loss = CrossEntropyLoss()
        
        # Metrics
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
    
    def forward(self, x):
        """Forward pass through student model."""
        return self.student(x)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return [optimizer], [scheduler]
    
    def compute_weights(
        self, 
        teacher_logits: List[torch.Tensor], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic weights for each teacher based on their CE loss.
        
        Uses exponential-based weighting scheme (Eq. 1):
        w_k = (1/(K-1)) * [1 - exp(L_CE^k) / sum_j exp(L_CE^j)]
        
        Args:
            teacher_logits: List of logits from each teacher
            labels: Ground truth labels
            
        Returns:
            Weights for each teacher, shape (batch_size, num_teachers)
        """
        K = len(teacher_logits)
        
        # Compute CE loss for each teacher (per-sample)
        ce_losses = []
        for logit in teacher_logits:
            ce = F.cross_entropy(logit, labels, reduction='none')
            ce_losses.append(ce)
        
        ce_losses = torch.stack(ce_losses, dim=1)  # (batch_size, K)
        
        # Compute exponential weights
        exp_losses = torch.exp(ce_losses)
        sum_exp_losses = torch.sum(exp_losses, dim=1, keepdim=True)
        
        weights = (1.0 / (K - 1)) * (1.0 - (exp_losses / sum_exp_losses))
        
        return weights
    
    def compute_average_teacher_confidence(
        self, 
        teacher_logits: List[torch.Tensor], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute average confidence that teachers have in the true class.
        
        Args:
            teacher_logits: List of logits from each teacher
            labels: Ground truth labels
            
        Returns:
            Average teacher confidence per sample
        """
        teacher_probs = []
        for logit in teacher_logits:
            probs = F.softmax(logit, dim=1)
            true_class_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            teacher_probs.append(true_class_probs)
        
        teacher_probs = torch.stack(teacher_probs, dim=1)
        avg_teacher_conf = torch.mean(teacher_probs, dim=1)
        
        return avg_teacher_conf
    
    def compute_losses(
        self, 
        student_logits: torch.Tensor, 
        labels: torch.Tensor, 
        teacher_logits: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total loss with soft and hard components using dynamic gating (Method 2).
        
        Formula (Eq. 5-6):
        L_total = α * L_KL + (1-α) * L_CE
        where α = sigmoid(γ * (Conf_avg - θ))
        
        Args:
            student_logits: Logits from student model
            labels: Ground truth labels
            teacher_logits: List of logits from teacher models
            
        Returns:
            Total loss value
        """
        loss_total = torch.tensor(0.0, device=self.device)
        log_dict = {}
        
        # Soft loss (KD)
        if self.use_soft_loss:
            weights = self.compute_weights(teacher_logits, labels)
            kd_loss = 0.0
            
            for k in range(self.num_teachers):
                z_T_k = teacher_logits[k]
                log_student = F.log_softmax(student_logits / self.temperature, dim=1)
                soft_teacher = F.softmax(z_T_k / self.temperature, dim=1)
                loss_kd_k = self.kl_div_loss(log_student, soft_teacher) * (self.temperature ** 2)
                weight_k = torch.mean(weights[:, k])
                kd_loss += weight_k * loss_kd_k
            
            log_dict['loss_soft'] = kd_loss
        
        # Hard loss (CE)
        if self.use_hard_loss:
            loss_hard = self.ce_loss(student_logits, labels)
            log_dict['loss_hard'] = loss_hard
        
        # Combine with gating (Eq. 5-6)
        if self.use_soft_loss and self.use_hard_loss:
            if self.num_teachers > 1:
                avg_teacher_conf = self.compute_average_teacher_confidence(
                    teacher_logits, labels
                )
                gate = torch.sigmoid(self.gamma * (avg_teacher_conf - self.threshold))
                gate_scalar = torch.mean(gate)
                
                loss_total = gate_scalar * kd_loss + (1 - gate_scalar) * (self.alpha * loss_hard)
                
                log_dict['gate'] = gate_scalar
                log_dict['avg_teacher_conf'] = avg_teacher_conf.mean()
            else:
                loss_total = self.alpha * log_dict['loss_soft'] + (1 - self.alpha) * log_dict['loss_hard']
                log_dict['gate'] = torch.tensor(1.0, device=self.device)
                log_dict['avg_teacher_conf'] = self.compute_average_teacher_confidence(
                    teacher_logits, labels
                ).mean()
        elif self.use_soft_loss:
            loss_total = log_dict['loss_soft']
            log_dict['gate'] = torch.tensor(1.0, device=self.device)
            log_dict['avg_teacher_conf'] = self.compute_average_teacher_confidence(
                teacher_logits, labels
            ).mean()
        elif self.use_hard_loss:
            loss_total = self.alpha * log_dict['loss_hard']
            log_dict['gate'] = torch.tensor(0.0, device=self.device)
            log_dict['avg_teacher_conf'] = torch.tensor(0.0, device=self.device)
        
        # Logging
        self.log_dict({
            'train/loss_soft': log_dict.get('loss_soft', torch.tensor(0.0, device=self.device)),
            'train/loss_hard': log_dict.get('loss_hard', torch.tensor(0.0, device=self.device)),
            'train/gate': log_dict.get('gate', torch.tensor(0.0, device=self.device)),
            'train/loss_total': loss_total,
            'train/avg_teacher_conf': log_dict.get('avg_teacher_conf', torch.tensor(0.0, device=self.device))
        }, on_epoch=True, prog_bar=True)
        
        return loss_total
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        teacher_logits = []
        with torch.no_grad():
            for i in range(self.num_teachers):
                teacher_images = batch[f'teacher_input_{i}'].to(self.device)
                logits = self.teachers[i](teacher_images)
                teacher_logits.append(logits)
        
        loss = self.compute_losses(student_logits, labels, teacher_logits)
        return loss
    
    def shared_eval_step(self, batch, stage: str):
        """Shared evaluation step for validation and test."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        teacher_logits = []
        with torch.no_grad():
            for i in range(self.num_teachers):
                teacher_images = batch[f'teacher_input_{i}'].to(self.device)
                logits = self.teachers[i](teacher_images)
                teacher_logits.append(logits)
        
        # Compute loss (similar to training)
        loss = self.compute_losses(student_logits, labels, teacher_logits)
        
        # Metrics
        preds = student_logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # AUROC
        if stage == 'val':
            self.val_auroc(student_logits, labels)
            self.log('val/auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        elif stage == 'test':
            self.test_auroc(student_logits, labels)
            self.log('test/auroc', self.test_auroc, on_epoch=True, prog_bar=True)
        
        self.log(f'{stage}/accuracy', acc, on_epoch=True, prog_bar=True)
        self.log(f'{stage}/loss_total', loss, on_epoch=True, prog_bar=True)
        
        return acc
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.shared_eval_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.shared_eval_step(batch, 'test')


class ConfidenceBasedKDLitModule(pl.LightningModule):
    """
    Method 3: Adaptive α-Guided KD (Confidence-Based Teacher Selection)
    
    This variant selects the most confident teacher for each sample
    and uses the teacher's confidence as the alpha weight.
    
    From the paper:
    - α_i = max_k p_Tk(y_i | x_i) (confidence of most confident teacher)
    - L_total = α * L_KL* + (1-α) * L_CE
    where L_KL* uses only the most confident teacher's logits
    """
    
    def __init__(
        self,
        teacher_models: List[nn.Module],
        student_model: nn.Module,
        temperature: float = 4.0,
        learning_rate: float = 1e-3,
        use_soft_loss: bool = True,
        use_hard_loss: bool = True,
        num_classes: int = 100
    ):
        """
        Initialize Confidence-Based KD Module (Method 3).
        
        Args:
            teacher_models: List of pre-trained teacher models
            student_model: Student model to be trained
            temperature: Temperature for softening probabilities
            learning_rate: Learning rate for optimizer
            use_soft_loss: Whether to use KL Divergence loss
            use_hard_loss: Whether to use Cross-Entropy loss
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=["teacher_models", "student_model"])
        
        self.use_soft_loss = use_soft_loss
        self.use_hard_loss = use_hard_loss
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        if not (self.use_soft_loss or self.use_hard_loss):
            raise ValueError("At least one of 'use_soft_loss' or 'use_hard_loss' must be True.")
        
        self.student = student_model
        
        if self.use_soft_loss:
            if not teacher_models or len(teacher_models) == 0:
                raise ValueError("Teacher models must be provided if 'use_soft_loss' is True.")
            self.teachers = teacher_models
            self.num_teachers = len(self.teachers)
            
            for teacher in self.teachers:
                teacher.eval()
                for param in teacher.parameters():
                    param.requires_grad = False
                    
            self.kl_div_loss = KLDivLoss(reduction='batchmean')
        else:
            self.teachers = None
            self.num_teachers = 0
        
        if self.use_hard_loss:
            self.ce_loss = CrossEntropyLoss()
        
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
    
    def forward(self, x):
        """Forward pass through student model."""
        return self.student(x)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return [optimizer], [scheduler]
    
    def get_most_confident_teacher_logits(
        self, 
        teacher_logits: List[torch.Tensor], 
        labels: torch.Tensor
    ):
        """
        Get logits from the most confident teacher for each sample.
        
        Selects teacher with highest p_Tk(y_i | x_i) for each sample.
        
        Args:
            teacher_logits: List of logits from each teacher
            labels: Ground truth labels
            
        Returns:
            Tuple of (best_teacher_logits, best_teacher_idx)
        """
        confidences_per_teacher = []
        for logit in teacher_logits:
            probs = F.softmax(logit, dim=1)
            correct_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            confidences_per_teacher.append(correct_prob)
        
        confidences_per_teacher = torch.stack(confidences_per_teacher, dim=1)
        max_conf_teacher_idx = torch.argmax(confidences_per_teacher, dim=1)
        
        batch_size = labels.size(0)
        out_logits = []
        for i in range(batch_size):
            idx = max_conf_teacher_idx[i]
            out_logits.append(teacher_logits[idx][i].unsqueeze(0))
        
        out_logits = torch.cat(out_logits, dim=0)
        return out_logits, max_conf_teacher_idx
    
    def compute_max_teacher_confidence(
        self, 
        teacher_logits: List[torch.Tensor], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute maximum teacher confidence for the correct label.
        
        Returns α_i = max_k p_Tk(y_i | x_i)
        
        Args:
            teacher_logits: List of logits from each teacher
            labels: Ground truth labels
            
        Returns:
            Maximum confidence per sample
        """
        teacher_probs = []
        for logit in teacher_logits:
            probs = F.softmax(logit, dim=1)
            true_class_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            teacher_probs.append(true_class_probs)
        
        teacher_probs = torch.stack(teacher_probs, dim=1)
        max_confidence, _ = torch.max(teacher_probs, dim=1)
        
        return max_confidence
    
    def compute_losses(
        self, 
        student_logits: torch.Tensor, 
        labels: torch.Tensor, 
        teacher_logits: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total loss with dynamic alpha based on teacher confidence (Method 3).
        
        L_total = α * L_KL* + (1-α) * L_CE
        where α = mean(max_k p_Tk(y_i | x_i))
        and L_KL* uses only the most confident teacher
        
        Args:
            student_logits: Logits from student model
            labels: Ground truth labels
            teacher_logits: List of logits from teacher models
            
        Returns:
            Total loss value
        """
        device = student_logits.device
        loss_total = torch.tensor(0.0, device=device)
        log_dict = {}
        
        if self.use_soft_loss:
            best_teacher_logits, best_teacher_idx = self.get_most_confident_teacher_logits(
                teacher_logits, labels
            )
            log_student = F.log_softmax(student_logits / self.temperature, dim=1)
            soft_teacher = F.softmax(best_teacher_logits / self.temperature, dim=1)
            kd_loss = self.kl_div_loss(log_student, soft_teacher) * (self.temperature ** 2)
            log_dict['loss_soft'] = kd_loss
            
            # Log teacher usage
            if self.num_teachers > 1:
                for i in range(self.num_teachers):
                    usage_fraction = (best_teacher_idx == i).float().mean()
                    self.log(
                        f"train/teacher_{i}_usage_fraction", 
                        usage_fraction, 
                        on_step=False, 
                        on_epoch=True
                    )
        else:
            kd_loss = torch.tensor(0.0, device=device)
        
        if self.use_hard_loss:
            loss_hard = self.ce_loss(student_logits, labels)
            log_dict['loss_hard'] = loss_hard
        else:
            loss_hard = torch.tensor(0.0, device=device)
        
        # Dynamic alpha based on confidence
        if self.use_soft_loss and self.use_hard_loss:
            max_conf = self.compute_max_teacher_confidence(teacher_logits, labels)
            alpha_scalar = max_conf.mean()
            loss_total = alpha_scalar * kd_loss + (1.0 - alpha_scalar) * loss_hard
            
            log_dict['dynamic_alpha'] = alpha_scalar
            log_dict['max_teacher_conf_mean'] = max_conf.mean()
        elif self.use_soft_loss:
            loss_total = kd_loss
            log_dict['dynamic_alpha'] = torch.tensor(1.0, device=device)
            log_dict['max_teacher_conf_mean'] = self.compute_max_teacher_confidence(
                teacher_logits, labels
            ).mean()
        elif self.use_hard_loss:
            loss_total = loss_hard
            log_dict['dynamic_alpha'] = torch.tensor(0.0, device=device)
            log_dict['max_teacher_conf_mean'] = torch.tensor(0.0, device=device)
        
        # Logging
        self.log_dict({
            'train/loss_soft': log_dict.get('loss_soft', torch.tensor(0.0, device=device)),
            'train/loss_hard': log_dict.get('loss_hard', torch.tensor(0.0, device=device)),
            'train/loss_total': loss_total,
            'train/dynamic_alpha': log_dict.get('dynamic_alpha', torch.tensor(0.0, device=device)),
            'train/max_teacher_conf_mean': log_dict.get('max_teacher_conf_mean', torch.tensor(0.0, device=device))
        }, on_epoch=True, prog_bar=True)
        
        return loss_total
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        teacher_logits = []
        with torch.no_grad():
            for i in range(self.num_teachers):
                teacher_images = batch[f'teacher_input_{i}'].to(self.device)
                logits = self.teachers[i](teacher_images)
                teacher_logits.append(logits)
        
        loss = self.compute_losses(student_logits, labels, teacher_logits)
        return loss
    
    def shared_eval_step(self, batch, stage: str):
        """Shared evaluation step."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        teacher_logits = []
        with torch.no_grad():
            for i in range(self.num_teachers):
                teacher_images = batch[f'teacher_input_{i}'].to(self.device)
                logits = self.teachers[i](teacher_images)
                teacher_logits.append(logits)
        
        loss = self.compute_losses(student_logits, labels, teacher_logits)
        
        preds = student_logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        if stage == 'val':
            self.val_auroc(student_logits, labels)
            self.log('val/auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        elif stage == 'test':
            self.test_auroc(student_logits, labels)
            self.log('test/auroc', self.test_auroc, on_epoch=True, prog_bar=True)
        
        self.log(f'{stage}/accuracy', acc, on_epoch=True, prog_bar=True)
        self.log(f'{stage}/loss_total', loss, on_epoch=True, prog_bar=True)
        
        return acc
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.shared_eval_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.shared_eval_step(batch, 'test')


class BaselineStudentModule(pl.LightningModule):
    """
    Baseline Student Module - Training from scratch without KD.
    
    This module trains a student model using only ground truth labels,
    without any knowledge distillation from teacher models.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        learning_rate: float = 1e-3,
        num_classes: int = 100
    ):
        """
        Initialize Baseline Student Module.
        
        Args:
            student_model: Student model to be trained
            learning_rate: Learning rate for optimizer
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=["student_model"])
        
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Student model
        if student_model is None:
            raise ValueError("A student model must be provided.")
        self.student = student_model
        
        # Loss function
        self.ce_loss = CrossEntropyLoss()
        
        # Metrics
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
    
    def forward(self, x):
        """Forward pass through student model."""
        return self.student(x)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        # Only CE loss
        loss = self.ce_loss(student_logits, labels)
        
        # Logging
        self.log('train/loss_total', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def shared_eval_step(self, batch, stage: str):
        """Shared evaluation step for validation and test."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        # Compute loss
        loss = self.ce_loss(student_logits, labels)
        
        # Metrics
        preds = student_logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # AUROC
        if stage == 'val':
            self.val_auroc(student_logits, labels)
            self.log('val/auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        elif stage == 'test':
            self.test_auroc(student_logits, labels)
            self.log('test/auroc', self.test_auroc, on_epoch=True, prog_bar=True)
        
        self.log(f'{stage}/accuracy', acc, on_epoch=True, prog_bar=True)
        self.log(f'{stage}/loss_total', loss, on_epoch=True, prog_bar=True)
        
        return acc
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.shared_eval_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.shared_eval_step(batch, 'test')
