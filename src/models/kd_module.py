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
        
        Special case: For single teacher (K=1), returns weight of 1.0
        
        Args:
            teacher_logits: List of logits from each teacher
            labels: Ground truth labels
            
        Returns:
            Weights for each teacher, shape (batch_size, num_teachers)
        """
        K = len(teacher_logits)
        batch_size = teacher_logits[0].shape[0]
        
        # Special case: single teacher - just return weight of 1.0
        if K == 1:
            return torch.ones((batch_size, 1), device=teacher_logits[0].device)
        
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
        
        Special case: For single teacher (K=1), returns weight of 1.0
        
        Args:
            teacher_logits: List of logits from each teacher
            labels: Ground truth labels
            
        Returns:
            Weights for each teacher, shape (batch_size, num_teachers)
        """
        K = len(teacher_logits)
        batch_size = teacher_logits[0].shape[0]
        
        # Special case: single teacher - just return weight of 1.0
        if K == 1:
            return torch.ones((batch_size, 1), device=teacher_logits[0].device)
        
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


class RegionAwareAttention(nn.Module):
    """
    Region-Aware Attention (RAA) Module from PAT Framework.
    
    Solves the VIEW MISMATCH problem between CNN students and ViT teachers.
    Transforms local CNN features into global representations using cross-stage attention.
    
    Reference: Lin et al. "Perspective-Aware Teaching: Adapting Knowledge for 
    Heterogeneous Distillation" arXiv:2501.08885
    https://github.com/jimmylin0979/PAT
    """
    
    def __init__(
        self,
        student_channels: List[int],  # Channels at each stage [96, 192, 384, 768] for example
        embed_dim: int = 256,
        num_heads: int = 8,
        patch_size: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize RAA module.
        
        Args:
            student_channels: List of channel dimensions at each CNN stage
            embed_dim: Dimension of the aligned embedding space
            num_heads: Number of attention heads
            patch_size: Size of patches to extract from feature maps
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_stages = len(student_channels)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Projection layers for each stage (project to shared embed_dim)
        self.stage_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )
            for ch in student_channels
        ])
        
        # Cross-stage self-attention
        self.cross_stage_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm for attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable stage embeddings
        self.stage_embeddings = nn.Parameter(torch.randn(1, self.num_stages, embed_dim) * 0.02)
    
    def patchify(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Convert feature map to patches (like ViT tokens).
        
        Pools to a fixed small spatial size first to prevent attention explosion
        when using real intermediate features (early stages can be 56×56).
        
        Args:
            feature_map: (B, C, H, W)
            
        Returns:
            patches: (B, num_patches, C)
        """
        B, C, H, W = feature_map.shape
        
        # Cap spatial size to avoid huge attention matrices
        # Without this, real features from early stages (56×56) create 784+ patches
        # per stage, leading to 1000+ total patches and O(n²) attention OOM.
        max_spatial = 7  # Pool to at most 7×7 before patchifying
        effective_h = min(H, max_spatial)
        
        target_size = max(effective_h // self.patch_size, 1)
        pooled = F.adaptive_avg_pool2d(feature_map, (target_size, target_size))
        
        # Flatten spatial dimensions
        patches = pooled.flatten(2).transpose(1, 2)  # (B, num_patches, C)
        
        return patches
    
    def forward(self, stage_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through RAA.
        
        Args:
            stage_features: List of feature maps from each CNN stage
                           [(B, C1, H1, W1), (B, C2, H2, W2), ...]
                           
        Returns:
            aligned_features: (B, embed_dim) - Global representation
        """
        B = stage_features[0].shape[0]
        
        all_patches = []
        
        # Process each stage
        for i, (feat, projector) in enumerate(zip(stage_features, self.stage_projectors)):
            # Pre-pool large feature maps BEFORE projection to save memory.
            # Without this, the 1x1 conv projector creates huge intermediate
            # activations (e.g. 256×56×56 for stage 0) that are immediately
            # thrown away by patchify pooling — 69× more memory for no benefit.
            if feat.shape[2] > 7 or feat.shape[3] > 7:
                feat = F.adaptive_avg_pool2d(feat, 7)
            
            # Project to shared dimension (now operates on ≤7×7 maps)
            projected = projector(feat)  # (B, embed_dim, ≤7, ≤7)
            
            # Patchify
            patches = self.patchify(projected)  # (B, num_patches, embed_dim)
            
            # Add stage embedding
            patches = patches + self.stage_embeddings[:, i:i+1, :]
            
            all_patches.append(patches)
        
        # Concatenate all patches from all stages
        all_patches = torch.cat(all_patches, dim=1)  # (B, total_patches, embed_dim)
        
        # Apply cross-stage self-attention
        normed = self.norm1(all_patches)
        attended, _ = self.cross_stage_attention(normed, normed, normed)
        all_patches = all_patches + attended
        
        # FFN
        all_patches = all_patches + self.ffn(self.norm2(all_patches))
        
        # Global pooling to get final representation
        aligned_features = all_patches.mean(dim=1)  # (B, embed_dim)
        
        return aligned_features


class AdaptiveFeedbackPrompt(nn.Module):
    """
    Adaptive Feedback Prompt (AFP) Module from PAT Framework.
    
    Solves the TEACHER UNAWARENESS problem by making the teacher adapt
    to the student's learning progress using prompt tuning.
    
    Reference: Lin et al. "Perspective-Aware Teaching: Adapting Knowledge for 
    Heterogeneous Distillation" arXiv:2501.08885
    """
    
    def __init__(
        self,
        teacher_dim: int,
        prompt_dim: int = 64,
        num_prompts: int = 4
    ):
        """
        Initialize AFP module.
        
        Args:
            teacher_dim: Dimension of teacher features
            prompt_dim: Dimension of prompt embeddings
            num_prompts: Number of prompt tokens
        """
        super().__init__()
        
        self.teacher_dim = teacher_dim
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        
        # Learnable prompt embeddings
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, prompt_dim) * 0.02)
        
        # Feedback encoder: encodes the error signal (T - S)
        self.feedback_encoder = nn.Sequential(
            nn.Linear(teacher_dim, prompt_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_dim * 2, prompt_dim)
        )
        
        # Prompt-to-feature projection
        self.prompt_projector = nn.Sequential(
            nn.Linear(prompt_dim, teacher_dim),
            nn.ReLU(inplace=True),
            nn.Linear(teacher_dim, teacher_dim)
        )
        
        # Gating mechanism to control adaptation strength
        self.gate = nn.Sequential(
            nn.Linear(teacher_dim + prompt_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        frozen_teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Adapt teacher features based on student feedback.
        
        Args:
            teacher_features: Current teacher features (B, D)
            student_features: Current student features (B, D_s) - may need projection
            frozen_teacher_features: Original frozen teacher features for regularization
            
        Returns:
            adapted_features: Teacher features adapted to student's needs (B, D)
        """
        B = teacher_features.shape[0]
        
        # Compute feedback (error residue)
        # Need to handle dimension mismatch between teacher and student
        if student_features.shape[-1] != teacher_features.shape[-1]:
            # Project student to teacher dimension for comparison
            feedback = teacher_features  # Use teacher as base
        else:
            feedback = teacher_features - student_features
        
        # Encode feedback
        feedback_encoded = self.feedback_encoder(feedback)  # (B, prompt_dim)
        
        # Combine with learnable prompts
        prompts_expanded = self.prompts.expand(B, -1, -1)  # (B, num_prompts, prompt_dim)
        
        # Add feedback to prompts
        feedback_expanded = feedback_encoded.unsqueeze(1)  # (B, 1, prompt_dim)
        modulated_prompts = prompts_expanded + feedback_expanded  # (B, num_prompts, prompt_dim)
        
        # Pool prompts
        prompt_pooled = modulated_prompts.mean(dim=1)  # (B, prompt_dim)
        
        # Project to teacher feature space
        adaptation = self.prompt_projector(prompt_pooled)  # (B, teacher_dim)
        
        # Compute gating weight
        gate_input = torch.cat([teacher_features, prompt_pooled], dim=-1)
        gate_weight = self.gate(gate_input)  # (B, 1)
        
        # Apply gated adaptation
        adapted_features = teacher_features + gate_weight * adaptation
        
        return adapted_features


class PATKDLitModule(pl.LightningModule):
    """
    PAT: Perspective-Aware Teaching Knowledge Distillation Framework.
    
    A universal KD framework for heterogeneous architectures (CNN ↔ ViT).
    
    Key Components:
    1. RAA (Region-Aware Attention): Solves VIEW MISMATCH by transforming
       student's local features into global representation via cross-stage attention.
    
    2. AFP (Adaptive Feedback Prompts): Solves TEACHER UNAWARENESS by making
       teacher adapt to student's learning progress using prompt tuning.
    
    Loss Function:
    L_PAT = L_CE + α·L_KL + β·L_FD + γ·L_Reg
    
    Reference: Lin et al. "Perspective-Aware Teaching: Adapting Knowledge for 
    Heterogeneous Distillation" arXiv:2501.08885
    https://github.com/jimmylin0979/PAT
    """
    
    def __init__(
        self,
        teacher_models: List[nn.Module],
        student_model: nn.Module,
        temperature: float = 4.0,
        learning_rate: float = 1e-3,
        alpha: float = 1.0,       # Weight for L_KL (logit distillation)
        beta: float = 1.0,        # Weight for L_FD (feature distillation)
        gamma: float = 0.1,       # Weight for L_Reg (regularization)
        num_classes: int = 100,
        student_channels: List[int] = None,  # [24, 32, 96, 1280] for MobileNetV2
        teacher_feature_dim: int = 768,      # 768 for ViT, 2048 for ResNet
        embed_dim: int = 256,
        num_heads: int = 8,
    ):
        """
        Initialize PAT-KD Module.
        
        Args:
            teacher_models: List of pre-trained teacher models (single teacher)
            student_model: Student model to be trained
            temperature: Temperature for softening probabilities
            learning_rate: Learning rate for optimizer
            alpha: Weight for KL divergence loss
            beta: Weight for feature distillation loss
            gamma: Weight for regularization loss
            num_classes: Number of output classes
            student_channels: Channel dims at each student stage
            teacher_feature_dim: Teacher's feature dimension
            embed_dim: Embedding dimension for RAA
            num_heads: Number of attention heads in RAA
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=["teacher_models", "student_model"])
        
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.teacher_feature_dim = teacher_feature_dim
        
        # Default student channels for MobileNetV2
        # Actual channels: block2=24, block5=32, block12=96, conv_head=1280
        if student_channels is None:
            student_channels = [24, 32, 96, 1280]
        self.student_channels = student_channels
        
        # Student model
        if student_model is None:
            raise ValueError("A student model must be provided.")
        self.student = student_model
        
        # Teacher model (frozen for main forward, but AFP adapts it)
        if not teacher_models or len(teacher_models) == 0:
            raise ValueError("Teacher model must be provided for PAT.")
        self.teacher = teacher_models[0]
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # RAA: Region-Aware Attention for student
        self.raa = RegionAwareAttention(
            student_channels=student_channels,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # AFP: Adaptive Feedback Prompts for teacher
        self.afp = AdaptiveFeedbackPrompt(
            teacher_dim=teacher_feature_dim,
            prompt_dim=embed_dim // 2
        )
        
        # Projection heads to align dimensions for feature distillation
        self.student_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, teacher_feature_dim)
        )
        
        # Loss functions
        self.kl_div_loss = KLDivLoss(reduction='batchmean')
        self.ce_loss = CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Metrics
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
    
    # ── checkpoint size fix: exclude frozen teacher ───────────────────
    # The teacher is re-loaded from HuggingFace at init; no need to save
    # its ~344MB (ViT) in every .ckpt file.

    def state_dict(self, *args, **kwargs):
        """Exclude frozen teacher weights from checkpoints."""
        full = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full.items() if not k.startswith('teacher.')}

    def load_state_dict(self, state_dict, strict=True):
        """Load checkpoint, tolerating missing teacher keys."""
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, x):
        """Forward pass through student model."""
        return self.student(x)
    
    def configure_optimizers(self):
        """Configure optimizer with warmup + cosine annealing.
        
        Warmup prevents gradient explosion at the start of training,
        which is critical when feature loss gradients flow through
        the student's early layers via real intermediate features.
        """
        # Trainable: student + RAA + AFP + projection (no channel_adapters needed)
        params = (
            list(self.student.parameters()) +
            list(self.raa.parameters()) +
            list(self.afp.parameters()) +
            list(self.student_proj.parameters())
        )
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Linear warmup for 5 epochs, then cosine decay
        warmup_epochs = 5
        max_epochs = 300
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4 / max(self.learning_rate, 1e-8),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        
        return [optimizer], [scheduler]
    
    def _student_forward_with_features(self, x: torch.Tensor):
        """
        Run student forward pass, capturing REAL intermediate features at each stage.
        
        Extracts actual features from different network depths instead of faking
        multi-scale features by pooling the final feature map.
        
        Returns:
            stage_features: List of feature tensors from each student stage
            logits: Student output logits
        """
        features = []
        
        if hasattr(self.student, 'blocks'):
            # ---- timm EfficientNet / MobileNetV2 ----
            # MobileNetV2 stage boundaries (flattened block indices):
            #   Stage 1: idx 2  -> (24,  H/4,  W/4)
            #   Stage 2: idx 5  -> (32,  H/8,  W/8)
            #   Stage 3: idx 12 -> (96,  H/16, W/16)
            #   Stage 4: after conv_head+bn2 -> (1280, H/32, W/32)
            target_indices = {2, 5, 12}
            
            x = self.student.conv_stem(x)
            x = self.student.bn1(x)
            if hasattr(self.student, 'act1'):
                x = self.student.act1(x)
            
            block_idx = 0
            for group in self.student.blocks:
                for block in group:
                    x = block(x)
                    if block_idx in target_indices:
                        features.append(x)
                    block_idx += 1
            
            # Stage 4: after conv_head + bn2
            x = self.student.conv_head(x)
            x = self.student.bn2(x)
            features.append(x)
            
            # Apply remaining activation before head
            if hasattr(self.student, 'act2'):
                x = self.student.act2(x)
            
            # Get logits through model's own head
            logits = self.student.forward_head(x)
            
            return features, logits
        
        elif hasattr(self.student, 'layer1'):
            # ---- timm ResNet ----
            x = self.student.conv1(x)
            x = self.student.bn1(x)
            x = self.student.act1(x)
            if hasattr(self.student, 'maxpool'):
                x = self.student.maxpool(x)
            
            x = self.student.layer1(x)
            features.append(x)
            x = self.student.layer2(x)
            features.append(x)
            x = self.student.layer3(x)
            features.append(x)
            x = self.student.layer4(x)
            features.append(x)
            
            logits = self.student.forward_head(x)
            return features, logits
        
        else:
            raise NotImplementedError(
                f"Cannot extract intermediate features from {type(self.student).__name__}. "
                f"Supported: EfficientNet/MobileNetV2 (timm), ResNet."
            )
    
    def extract_teacher_features(self, x: torch.Tensor):
        """
        Extract features and logits from teacher.
        
        Args:
            x: Input images
            
        Returns:
            features: Teacher features (B, D)
            logits: Teacher logits (B, num_classes)
        """
        with torch.no_grad():
            if hasattr(self.teacher, 'forward_features'):
                features = self.teacher.forward_features(x)
                
                # Handle different feature shapes
                if len(features.shape) == 4:  # CNN: (B, C, H, W)
                    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
                elif len(features.shape) == 3:  # ViT: (B, N, D)
                    features = features[:, 0]  # CLS token
                
                # Get logits
                if hasattr(self.teacher, 'head'):
                    logits = self.teacher.head(features)
                elif hasattr(self.teacher, 'classifier'):
                    logits = self.teacher.classifier(features)
                elif hasattr(self.teacher, 'fc'):
                    logits = self.teacher.fc(features)
                else:
                    logits = self.teacher(x)
            elif hasattr(self.teacher, 'features') and hasattr(self.teacher, 'classifier'):
                # Handle torchvision DenseNet (and similar models)
                # DenseNet structure: features (Sequential) -> ReLU -> Pool -> classifier
                features = self.teacher.features(x)
                features = F.relu(features, inplace=False)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                logits = self.teacher.classifier(features)
            else:
                logits = self.teacher(x)
                features = logits
        
        return features, logits
    
    def compute_pat_loss(
        self,
        student_logits: torch.Tensor,
        student_aligned: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_adapted: torch.Tensor,
        frozen_teacher_features: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Compute PAT loss with all components.
        
        L_PAT = L_CE + α·L_KL + β·L_FD + γ·L_Reg
        
        Args:
            student_logits: Student's output logits
            student_aligned: Student features after RAA alignment
            teacher_logits: Teacher's output logits  
            teacher_features: Teacher's features
            teacher_adapted: Teacher features after AFP adaptation
            frozen_teacher_features: Original frozen teacher features
            labels: Ground truth labels
            
        Returns:
            total_loss, log_dict
        """
        log_dict = {}
        
        # 1. L_CE: Cross-entropy with ground truth
        loss_ce = self.ce_loss(student_logits, labels)
        log_dict['loss_ce'] = loss_ce
        
        # 2. L_KL: Soft logit distillation
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        loss_kl = self.kl_div_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        log_dict['loss_kl'] = loss_kl
        
        # 3. L_FD: Feature distillation (RAA-student vs AFP-teacher)
        # Project student aligned features to teacher dimension
        student_proj = self.student_proj(student_aligned)
        
        # Normalize for stable comparison
        student_proj_norm = F.normalize(student_proj, p=2, dim=1)
        teacher_adapted_norm = F.normalize(teacher_adapted, p=2, dim=1)
        
        loss_fd = self.mse_loss(student_proj_norm, teacher_adapted_norm)
        log_dict['loss_fd'] = loss_fd
        
        # 4. L_Reg: Regularization to anchor adapted teacher to frozen teacher
        frozen_norm = F.normalize(frozen_teacher_features, p=2, dim=1)
        adapted_norm = F.normalize(teacher_adapted, p=2, dim=1)
        loss_reg = self.mse_loss(adapted_norm, frozen_norm)
        log_dict['loss_reg'] = loss_reg
        
        # Total loss
        loss_total = (
            loss_ce + 
            self.alpha * loss_kl + 
            self.beta * loss_fd + 
            self.gamma * loss_reg
        )
        log_dict['loss_total'] = loss_total
        
        return loss_total, log_dict
    
    def training_step(self, batch, batch_idx):
        """Training step — single student forward pass for both features and logits."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        teacher_images = batch['teacher_input_0'].to(self.device)
        
        # 1. Single student forward: extract REAL intermediate features + logits in ONE pass
        student_stage_features, student_logits = self._student_forward_with_features(student_images)
        
        # 2. Detach early-stage features to prevent feature loss gradients from
        #    destabilizing early student layers. Only later stages (which have
        #    higher-level representations) pass gradients through the RAA path.
        #    Early layers still get gradients from CE and KL losses via logits.
        #    stages: [block2(24ch), block5(32ch), block12(96ch), conv_head(1280ch)]
        #    Detach stage 0 and 1 (low-level edges/textures), keep 2 and 3 (semantics)
        stable_features = [
            feat.detach() if i < 2 else feat
            for i, feat in enumerate(student_stage_features)
        ]
        
        # 3. Apply RAA to get aligned student representation
        student_aligned = self.raa(stable_features)  # (B, embed_dim)
        
        # 3. Extract teacher features (frozen)
        teacher_features, teacher_logits = self.extract_teacher_features(teacher_images)
        frozen_teacher_features = teacher_features.clone()
        
        # 4. Apply AFP to adapt teacher features
        student_proj_for_feedback = self.student_proj(student_aligned)
        teacher_adapted = self.afp(
            teacher_features,
            student_proj_for_feedback,
            frozen_teacher_features
        )
        
        # 5. Compute PAT loss
        loss_total, log_dict = self.compute_pat_loss(
            student_logits=student_logits,
            student_aligned=student_aligned,
            teacher_logits=teacher_logits,
            teacher_features=teacher_features,
            teacher_adapted=teacher_adapted,
            frozen_teacher_features=frozen_teacher_features,
            labels=labels
        )
        
        # Logging
        self.log_dict({
            'train/loss_ce': log_dict['loss_ce'],
            'train/loss_kl': log_dict['loss_kl'],
            'train/loss_fd': log_dict['loss_fd'],
            'train/loss_reg': log_dict['loss_reg'],
            'train/loss_total': loss_total,
        }, on_epoch=True, prog_bar=True)
        
        return loss_total
    
    def shared_eval_step(self, batch, stage: str):
        """Shared evaluation step for validation and test."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        
        # Get student predictions (no RAA/AFP needed for inference)
        student_logits = self.student(student_images)
        
        # CE loss only for evaluation
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


# ═══════════════════════════════════════════════════════════════════════
# Multi-Teacher PAT Methods (Methods A, B, D)
# Extends PAT from single-teacher to multi-teacher knowledge distillation
# ═══════════════════════════════════════════════════════════════════════


class TeacherFusionAttention(nn.Module):
    """
    Cross-attention module for fusing multiple teachers' adapted features.

    The student's RAA representation acts as Query, while each teacher's
    AFP-adapted features act as Keys/Values.  The student learns to
    dynamically attend to the most relevant teacher for each sample.

    Used by PAT-Attn (Method D).
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dims: List[int],
        attn_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            student_dim: Dimension of student RAA features (embed_dim)
            teacher_dims: List of per-teacher feature dimensions [D_1, ..., D_K]
            attn_dim: Shared attention dimension d_a
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.attn_dim = attn_dim
        self.num_teachers = len(teacher_dims)

        # Query projection from student RAA space
        self.W_Q = nn.Linear(student_dim, attn_dim)

        # Per-teacher Key/Value projections (teacher_dim -> attn_dim)
        self.W_K = nn.ModuleList([
            nn.Linear(td, attn_dim) for td in teacher_dims
        ])
        self.W_V = nn.ModuleList([
            nn.Linear(td, attn_dim) for td in teacher_dims
        ])

        # Multi-head cross-attention (query: 1 token, key/value: K tokens)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(attn_dim)

        # Output projection back to student space for feature distillation
        self.out_proj = nn.Linear(attn_dim, attn_dim)

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features_list: List[torch.Tensor],
    ):
        """
        Fuse multiple teacher features via cross-attention.

        Args:
            student_features: Student RAA output (B, student_dim)
            teacher_features_list: List of AFP-adapted teacher features
                                   [(B, D_1), ..., (B, D_K)]

        Returns:
            fused_features: (B, attn_dim)
            attn_weights: (B, K) — per-sample attention over teachers
        """
        B = student_features.shape[0]

        # Query: (B, 1, attn_dim)
        Q = self.W_Q(student_features).unsqueeze(1)

        # Keys & Values: project each teacher, stack → (B, K, attn_dim)
        K_list = [self.W_K[k](teacher_features_list[k]) for k in range(self.num_teachers)]
        V_list = [self.W_V[k](teacher_features_list[k]) for k in range(self.num_teachers)]

        K = torch.stack(K_list, dim=1)  # (B, K, attn_dim)
        V = torch.stack(V_list, dim=1)  # (B, K, attn_dim)

        # Cross-attention: student queries teachers
        attn_out, attn_weights_raw = self.cross_attn(Q, K, V)
        # attn_out: (B, 1, attn_dim), attn_weights_raw: (B, 1, K)

        fused = self.norm(attn_out.squeeze(1))  # (B, attn_dim)
        fused = self.out_proj(fused)

        # Attention weights per teacher (B, K)
        attn_weights = attn_weights_raw.squeeze(1)

        return fused, attn_weights


class MultiTeacherPATIndLitModule(pl.LightningModule):
    """
    Method A: PAT-Ind — Independent multi-teacher PAT.

    Runs a separate AFP + projection pathway per teacher while sharing
    the student-side RAA.  All teacher losses are averaged equally.

    L = L_CE + (1/K) * Σ_k [α·L_KL_k + β·L_FD_k + γ·L_Reg_k]
    """

    def __init__(
        self,
        teacher_models: List[nn.Module],
        student_model: nn.Module,
        temperature: float = 4.0,
        learning_rate: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        num_classes: int = 100,
        student_channels: List[int] = None,
        teacher_feature_dims: List[int] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher_models", "student_model"])

        self.temperature = temperature
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        if student_channels is None:
            student_channels = [24, 32, 96, 1280]
        self.student_channels = student_channels

        # Student
        if student_model is None:
            raise ValueError("A student model must be provided.")
        self.student = student_model

        # Teachers (frozen)
        if not teacher_models or len(teacher_models) == 0:
            raise ValueError("At least one teacher model must be provided.")
        self.teachers = nn.ModuleList(teacher_models)
        self.num_teachers = len(teacher_models)
        for t in self.teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad = False

        if teacher_feature_dims is None:
            teacher_feature_dims = [768] * self.num_teachers
        self.teacher_feature_dims = teacher_feature_dims

        # Shared RAA (student side)
        self.raa = RegionAwareAttention(
            student_channels=student_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        # Per-teacher AFP modules
        self.afps = nn.ModuleList([
            AdaptiveFeedbackPrompt(
                teacher_dim=td,
                prompt_dim=embed_dim // 2,
            )
            for td in teacher_feature_dims
        ])

        # Per-teacher projection heads (embed_dim -> teacher_dim)
        self.student_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, td),
            )
            for td in teacher_feature_dims
        ])

        # Loss functions
        self.kl_div_loss = KLDivLoss(reduction='batchmean')
        self.ce_loss = CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # Metrics
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')

    # ── checkpoint size fix: exclude frozen teachers ────────────────
    # Without this, every .ckpt includes all 3 frozen teachers (~464MB)
    # making checkpoints ~500MB instead of ~34MB.  With WandB log_model='all'
    # keeping local copies, this fills the 32GB root overlay in <100 epochs.

    def state_dict(self, *args, **kwargs):
        """Exclude frozen teacher weights from checkpoints."""
        full = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full.items() if not k.startswith('teachers.')}

    def load_state_dict(self, state_dict, strict=True):
        """Load checkpoint, tolerating missing teacher keys (they're re-loaded at init)."""
        return super().load_state_dict(state_dict, strict=False)

    # ── forward / optimizers ──────────────────────────────────────────

    def forward(self, x):
        return self.student(x)

    def configure_optimizers(self):
        params = (
            list(self.student.parameters()) +
            list(self.raa.parameters()) +
            list(self.afps.parameters()) +
            list(self.student_projs.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=1e-4)

        warmup_epochs = 5
        max_epochs = 300
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4 / max(self.learning_rate, 1e-8),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        return [optimizer], [scheduler]

    # ── student feature extraction ────────────────────────────────────

    def _student_forward_with_features(self, x: torch.Tensor):
        """Run student forward pass, capturing real intermediate features."""
        features = []
        if hasattr(self.student, 'blocks'):
            target_indices = {2, 5, 12}
            x = self.student.conv_stem(x)
            x = self.student.bn1(x)
            if hasattr(self.student, 'act1'):
                x = self.student.act1(x)
            block_idx = 0
            for group in self.student.blocks:
                for block in group:
                    x = block(x)
                    if block_idx in target_indices:
                        features.append(x)
                    block_idx += 1
            x = self.student.conv_head(x)
            x = self.student.bn2(x)
            features.append(x)
            if hasattr(self.student, 'act2'):
                x = self.student.act2(x)
            logits = self.student.forward_head(x)
            return features, logits
        elif hasattr(self.student, 'layer1'):
            x = self.student.conv1(x)
            x = self.student.bn1(x)
            x = self.student.act1(x)
            if hasattr(self.student, 'maxpool'):
                x = self.student.maxpool(x)
            for layer in [self.student.layer1, self.student.layer2,
                          self.student.layer3, self.student.layer4]:
                x = layer(x)
                features.append(x)
            logits = self.student.forward_head(x)
            return features, logits
        else:
            raise NotImplementedError(
                f"Cannot extract features from {type(self.student).__name__}."
            )

    # ── teacher feature extraction ────────────────────────────────────

    def _extract_teacher_features(self, x: torch.Tensor, teacher: nn.Module):
        """Extract penultimate features + logits from a single teacher."""
        with torch.no_grad():
            if hasattr(teacher, 'forward_features'):
                features = teacher.forward_features(x)
                if len(features.shape) == 4:
                    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
                elif len(features.shape) == 3:
                    features = features[:, 0]
                if hasattr(teacher, 'head'):
                    logits = teacher.head(features)
                elif hasattr(teacher, 'classifier'):
                    logits = teacher.classifier(features)
                elif hasattr(teacher, 'fc'):
                    logits = teacher.fc(features)
                else:
                    logits = teacher(x)
            elif hasattr(teacher, 'features') and hasattr(teacher, 'classifier'):
                features = teacher.features(x)
                features = F.relu(features, inplace=False)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                logits = teacher.classifier(features)
            else:
                logits = teacher(x)
                features = logits
        return features, logits

    # ── loss computation ──────────────────────────────────────────────

    def _compute_per_teacher_losses(
        self,
        student_logits, student_aligned, labels,
        teacher_features_k, teacher_logits_k,
        frozen_teacher_features_k, afp_k, proj_k,
    ):
        """Compute PAT losses for a single teacher channel."""
        # L_KL
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits_k / self.temperature, dim=1)
        loss_kl = self.kl_div_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # Project student features to this teacher's dim
        student_proj = proj_k(student_aligned)

        # AFP adaptation
        teacher_adapted = afp_k(
            teacher_features_k, student_proj, frozen_teacher_features_k,
        )

        # L_FD
        s_norm = F.normalize(student_proj, p=2, dim=1)
        t_norm = F.normalize(teacher_adapted, p=2, dim=1)
        loss_fd = self.mse_loss(s_norm, t_norm)

        # L_Reg
        frozen_norm = F.normalize(frozen_teacher_features_k, p=2, dim=1)
        adapted_norm = F.normalize(teacher_adapted, p=2, dim=1)
        loss_reg = self.mse_loss(adapted_norm, frozen_norm)

        return loss_kl, loss_fd, loss_reg, teacher_adapted

    # ── training / eval steps ─────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)

        # Student forward
        stage_feats, student_logits = self._student_forward_with_features(student_images)
        stable_feats = [f.detach() if i < 2 else f for i, f in enumerate(stage_feats)]
        student_aligned = self.raa(stable_feats)

        # L_CE
        loss_ce = self.ce_loss(student_logits, labels)

        total_kl = 0.0
        total_fd = 0.0
        total_reg = 0.0

        for k in range(self.num_teachers):
            t_images = batch[f'teacher_input_{k}'].to(self.device)
            t_feats, t_logits = self._extract_teacher_features(t_images, self.teachers[k])
            frozen_feats = t_feats.clone()

            kl, fd, reg, _ = self._compute_per_teacher_losses(
                student_logits, student_aligned, labels,
                t_feats, t_logits, frozen_feats,
                self.afps[k], self.student_projs[k],
            )
            total_kl = total_kl + kl
            total_fd = total_fd + fd
            total_reg = total_reg + reg

        # Average across teachers
        total_kl = total_kl / self.num_teachers
        total_fd = total_fd / self.num_teachers
        total_reg = total_reg / self.num_teachers

        loss_total = (
            loss_ce
            + self.alpha * total_kl
            + self.beta * total_fd
            + self.gamma * total_reg
        )

        self.log_dict({
            'train/loss_ce': loss_ce,
            'train/loss_kl': total_kl,
            'train/loss_fd': total_fd,
            'train/loss_reg': total_reg,
            'train/loss_total': loss_total,
        }, on_epoch=True, prog_bar=True)
        return loss_total

    def shared_eval_step(self, batch, stage: str):
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        loss = self.ce_loss(student_logits, labels)
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
        return self.shared_eval_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, 'test')


class MultiTeacherPATCWLitModule(MultiTeacherPATIndLitModule):
    """
    Method B: PAT-CW — Confidence-Weighted multi-teacher PAT.

    Extends PAT-Ind by weighting each teacher's contribution using
    CA-WKD per-sample confidence weights (from Methods 1-2).

    L = L_CE + Σ_k w̄_k [α·L_KL_k + β·L_FD_k + γ·L_Reg_k]

    where w_k = (1/(K-1)) * (1 - exp(CE_k) / Σ_j exp(CE_j))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_teacher_weights(
        self,
        teacher_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CA-WKD per-sample teacher weights.

        Returns:
            weights: (B, K) tensor of per-sample teacher weights.
        """
        K = len(teacher_logits)
        B = teacher_logits[0].shape[0]

        if K == 1:
            return torch.ones((B, 1), device=teacher_logits[0].device)

        ce_losses = torch.stack([
            F.cross_entropy(tl, labels, reduction='none') for tl in teacher_logits
        ], dim=1)  # (B, K)

        exp_losses = torch.exp(ce_losses)
        sum_exp = exp_losses.sum(dim=1, keepdim=True)
        weights = (1.0 / (K - 1)) * (1.0 - exp_losses / sum_exp)
        return weights

    def training_step(self, batch, batch_idx):
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)

        # Student forward
        stage_feats, student_logits = self._student_forward_with_features(student_images)
        stable_feats = [f.detach() if i < 2 else f for i, f in enumerate(stage_feats)]
        student_aligned = self.raa(stable_feats)

        # L_CE
        loss_ce = self.ce_loss(student_logits, labels)

        # Gather all teacher logits for weight computation
        all_teacher_feats = []
        all_teacher_logits = []
        all_frozen_feats = []
        for k in range(self.num_teachers):
            t_images = batch[f'teacher_input_{k}'].to(self.device)
            t_feats, t_logits = self._extract_teacher_features(t_images, self.teachers[k])
            all_teacher_feats.append(t_feats)
            all_teacher_logits.append(t_logits)
            all_frozen_feats.append(t_feats.clone())

        # Compute CA-WKD weights (B, K)
        weights = self._compute_teacher_weights(all_teacher_logits, labels)

        total_kl = 0.0
        total_fd = 0.0
        total_reg = 0.0

        for k in range(self.num_teachers):
            kl, fd, reg, _ = self._compute_per_teacher_losses(
                student_logits, student_aligned, labels,
                all_teacher_feats[k], all_teacher_logits[k],
                all_frozen_feats[k],
                self.afps[k], self.student_projs[k],
            )
            w_k = weights[:, k].mean()  # batch-averaged weight
            total_kl = total_kl + w_k * kl
            total_fd = total_fd + w_k * fd
            total_reg = total_reg + w_k * reg

        loss_total = (
            loss_ce
            + self.alpha * total_kl
            + self.beta * total_fd
            + self.gamma * total_reg
        )

        # Log per-teacher weights
        for k in range(self.num_teachers):
            self.log(f'train/teacher_{k}_weight', weights[:, k].mean(),
                     on_epoch=True, prog_bar=False)

        self.log_dict({
            'train/loss_ce': loss_ce,
            'train/loss_kl': total_kl,
            'train/loss_fd': total_fd,
            'train/loss_reg': total_reg,
            'train/loss_total': loss_total,
        }, on_epoch=True, prog_bar=True)
        return loss_total


class MultiTeacherPATAttnLitModule(MultiTeacherPATIndLitModule):
    """
    Method D: PAT-Attn — Attention-Based Teacher Fusion.

    Uses a learned cross-attention mechanism where the student's RAA
    features act as Query and each teacher's AFP-adapted features act
    as Keys/Values.  The student dynamically attends to the most
    relevant teacher per sample.

    L = L_CE + α·L_KL^attn + β·L_FD^attn + γ·(1/K)·Σ L_Reg_k + λ_e·L_entropy
    """

    def __init__(
        self,
        teacher_models: List[nn.Module],
        student_model: nn.Module,
        temperature: float = 4.0,
        learning_rate: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        num_classes: int = 100,
        student_channels: List[int] = None,
        teacher_feature_dims: List[int] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dim: int = 256,
        attn_heads: int = 4,
        entropy_weight: float = 0.01,
    ):
        # Store attn-specific params before super().__init__
        self._attn_dim = attn_dim
        self._attn_heads = attn_heads
        self._entropy_weight = entropy_weight
        self._teacher_feature_dims_init = teacher_feature_dims

        super().__init__(
            teacher_models=teacher_models,
            student_model=student_model,
            temperature=temperature,
            learning_rate=learning_rate,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            num_classes=num_classes,
            student_channels=student_channels,
            teacher_feature_dims=teacher_feature_dims,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.entropy_weight = entropy_weight
        self.attn_dim = attn_dim

        # Teacher Fusion Attention
        self.teacher_fusion = TeacherFusionAttention(
            student_dim=embed_dim,
            teacher_dims=self.teacher_feature_dims,
            attn_dim=attn_dim,
            num_heads=attn_heads,
        )

        # Student output projection for feature distillation (embed_dim -> attn_dim)
        self.student_out_proj = nn.Linear(embed_dim, attn_dim)

    def configure_optimizers(self):
        params = (
            list(self.student.parameters()) +
            list(self.raa.parameters()) +
            list(self.afps.parameters()) +
            list(self.student_projs.parameters()) +
            list(self.teacher_fusion.parameters()) +
            list(self.student_out_proj.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=1e-4)

        warmup_epochs = 5
        max_epochs = 300
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4 / max(self.learning_rate, 1e-8),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)

        # Student forward
        stage_feats, student_logits = self._student_forward_with_features(student_images)
        stable_feats = [f.detach() if i < 2 else f for i, f in enumerate(stage_feats)]
        student_aligned = self.raa(stable_feats)  # (B, embed_dim)

        # L_CE
        loss_ce = self.ce_loss(student_logits, labels)

        # Per-teacher AFP adaptation
        adapted_teacher_feats = []
        all_teacher_logits = []
        total_reg = 0.0

        for k in range(self.num_teachers):
            t_images = batch[f'teacher_input_{k}'].to(self.device)
            t_feats, t_logits = self._extract_teacher_features(t_images, self.teachers[k])
            frozen_feats = t_feats.clone()
            all_teacher_logits.append(t_logits)

            # Project student → teacher dim for AFP feedback
            student_proj_k = self.student_projs[k](student_aligned)
            teacher_adapted_k = self.afps[k](t_feats, student_proj_k, frozen_feats)
            adapted_teacher_feats.append(teacher_adapted_k)

            # L_Reg per teacher
            frozen_norm = F.normalize(frozen_feats, p=2, dim=1)
            adapted_norm = F.normalize(teacher_adapted_k, p=2, dim=1)
            total_reg = total_reg + self.mse_loss(adapted_norm, frozen_norm)

        total_reg = total_reg / self.num_teachers

        # ── Cross-attention fusion ────────────────────────────────────
        fused_teacher, attn_weights = self.teacher_fusion(
            student_aligned, adapted_teacher_feats,
        )  # fused: (B, attn_dim), attn_weights: (B, K)

        # L_FD: MSE between projected student and fused teacher
        student_proj_out = self.student_out_proj(student_aligned)
        s_norm = F.normalize(student_proj_out, p=2, dim=1)
        t_norm = F.normalize(fused_teacher, p=2, dim=1)
        loss_fd = self.mse_loss(s_norm, t_norm)

        # L_KL: attention-weighted logit distillation
        # z_T^attn = Σ A_k * z_T_k
        stacked_logits = torch.stack(all_teacher_logits, dim=1)  # (B, K, C)
        attn_expanded = attn_weights.unsqueeze(-1)  # (B, K, 1)
        merged_logits = (stacked_logits * attn_expanded).sum(dim=1)  # (B, C)

        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(merged_logits / self.temperature, dim=1)
        loss_kl = self.kl_div_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # L_entropy: encourage attention spread (prevent collapse)
        # H = -Σ A_k log(A_k); we MAXIMISE entropy → minimise -H
        eps = 1e-8
        loss_entropy = (attn_weights * torch.log(attn_weights + eps)).sum(dim=1).mean()
        # loss_entropy is negative when entropy is high; we add it so minimising
        # the total loss maximises entropy.

        loss_total = (
            loss_ce
            + self.alpha * loss_kl
            + self.beta * loss_fd
            + self.gamma * total_reg
            + self.entropy_weight * loss_entropy
        )

        # Logging
        log_dict = {
            'train/loss_ce': loss_ce,
            'train/loss_kl': loss_kl,
            'train/loss_fd': loss_fd,
            'train/loss_reg': total_reg,
            'train/loss_entropy': loss_entropy,
            'train/loss_total': loss_total,
        }
        for k in range(self.num_teachers):
            log_dict[f'train/attn_teacher_{k}'] = attn_weights[:, k].mean()

        self.log_dict(log_dict, on_epoch=True, prog_bar=True)
        return loss_total


def _hcd_remove_one_hot(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Mask out the ground-truth class position before computing orthogonality.
    Zeroes out the label position so normalization operates only on
    non-ground-truth dimensions.
    
    Args:
        logits: (B, k, C) sub-logits
        labels: (B,) ground truth labels
        
    Returns:
        Masked logits with ground-truth position zeroed out
    """
    B, k, C = logits.shape
    mask = torch.zeros_like(logits)
    mask.scatter_(2, labels.unsqueeze(1).unsqueeze(2).expand(B, k, 1), 1)
    # Zero out the ground-truth class so it doesn't affect normalization
    masked_logits = logits * (1 - mask)
    return masked_logits


def _hcd_orthogonality_loss(vectors: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute orthogonality loss among sub-logit vectors.
    Uses squared penalty on excess similarity above threshold.
    
    Reference: Official HCD implementation
    
    Args:
        vectors: (B, k, C) normalized sub-logits
        threshold: Similarity threshold (default 0.5)
        
    Returns:
        Orthogonality loss scalar
    """
    B, k, C = vectors.shape
    vectors = F.normalize(vectors, p=2, dim=-1)
    
    # Compute pairwise dot products
    dot_product = torch.einsum('bik,bjk->bij', vectors, vectors)  # (B, k, k)
    
    # Extract off-diagonal elements
    mask = torch.eye(k, device=vectors.device).bool()
    off_diagonal = dot_product[:, ~mask].view(B, k, k - 1)  # (B, k, k-1)
    
    # Squared penalty on excess similarity
    excess_sim = torch.relu(off_diagonal - threshold)
    loss = torch.mean(excess_sim.pow(2))
    return loss


class HCDKDLitModule(pl.LightningModule):
    """
    HCD: Heterogeneous Complementary Distillation (feature-based KD).
    
    Official implementation adapted from: https://github.com/yema-web/HCD
    Reference: Xu et al. "Heterogeneous Complementary Distillation" arXiv:2511.10942
    
    Loss formulation:
        loss_total = loss_gt + loss_hcd + loss_orthogonality
        
    Where:
        loss_gt = gt_loss_weight * (CE(student, labels) + sum(CE(fused_sublogits, labels)))
        loss_hcd = hcd_loss_weight * sum(KL(student || fused_sublogits))
        loss_orthogonality = diversity * sum(orthogonality_loss(masked_sublogits))
    """
    
    def __init__(
        self,
        teacher_models: List[nn.Module],
        student_model: nn.Module,
        temperature: float = 1.0,            # ofa_temperature in official
        learning_rate: float = 0.05,         # SGD lr in official
        hcd_loss_weight: float = 6.0,        # Weight for HCD KL loss
        kd_loss_weight: float = 1.0,         # Weight for standard KD loss (student vs teacher)
        gt_loss_weight: float = 1.0,         # Weight for CE losses
        diversity: float = 1.0,              # Weight for orthogonality loss
        num_classes: int = 100,
        student_channels: List[int] = None,  # Stage channels for student
        student_final_dim: int = 1280,       # Final student feature dim
        teacher_feature_dim: int = 768,      # Penultimate teacher feature dim
        embed_dim: int = 256,                # Projected stage feature dim (max(feat_s, feat_t))
        k: int = 4,                          # Number of sub-logits (hardcoded in official)
        ortho_threshold: float = 0.5,        # Orthogonality threshold
        lambda_student: float = 1.0,         # Weight for sub-logits in fusion
        lambda_teacher: float = 1.0,         # Weight for teacher logits in fusion
        label_smoothing: float = 0.1,        # Label smoothing (official uses 0.1)
    ):
        """
        Initialize HCD module matching official implementation.
        
        Args:
            teacher_models: List of pre-trained teacher models (single teacher)
            student_model: Student model to be trained
            temperature: Temperature for KL softmax (ofa_temperature)
            learning_rate: Learning rate for optimizer
            hcd_loss_weight: Weight for HCD KL distillation loss
            gt_loss_weight: Weight for ground-truth CE losses
            diversity: Weight for orthogonality loss
            num_classes: Number of output classes
            student_channels: Channel dims at each student stage
            student_final_dim: Final feature dimension of the student
            teacher_feature_dim: Teacher's penultimate feature dimension
            embed_dim: Projection dimension for student stage features
            k: Number of sub-logits for SDD (default 4)
            ortho_threshold: Threshold for orthogonality loss
            lambda_student: Weight for sub-logits in fusion
            lambda_teacher: Weight for teacher logits in fusion
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=["teacher_models", "student_model"])
        
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.hcd_loss_weight = hcd_loss_weight
        self.kd_loss_weight = kd_loss_weight
        self.gt_loss_weight = gt_loss_weight
        self.diversity = diversity
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.k = k
        self.ortho_threshold = ortho_threshold
        self.lambda_student = lambda_student
        self.lambda_teacher = lambda_teacher
        self.teacher_feature_dim = teacher_feature_dim
        self.student_final_dim = student_final_dim
        self.label_smoothing = label_smoothing
        
        if student_channels is None:
            student_channels = [24, 32, 96, 1280]  # MobileNetV2 real stage channels
        self.student_channels = student_channels
        self.num_stages = len(student_channels)
        
        if student_model is None:
            raise ValueError("A student model must be provided.")
        self.student = student_model
        
        if not teacher_models or len(teacher_models) == 0:
            raise ValueError("Teacher model must be provided for HCD.")
        self.teacher = teacher_models[0]
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Per-stage projectors: project student features to embed_dim
        # Matching official: SepConv blocks for CNN, then pool to vector
        self.stage_projectors = nn.ModuleList()
        for i, ch in enumerate(student_channels):
            if i < len(student_channels) - 1:
                # Earlier stages: need more downsampling
                down_sample_num = len(student_channels) - 1 - i
                layers = []
                in_ch = ch
                for j in range(down_sample_num):
                    if j == down_sample_num - 1:
                        out_ch = max(self.student_final_dim, teacher_feature_dim)
                    else:
                        out_ch = in_ch * 2
                    layers.append(self._make_sep_conv(in_ch, out_ch))
                    in_ch = out_ch
                layers.append(nn.AdaptiveAvgPool2d(1))
                layers.append(nn.Flatten())
                self.stage_projectors.append(nn.Sequential(*layers))
            else:
                # Final stage: just 1x1 conv + pool
                self.stage_projectors.append(nn.Sequential(
                    nn.Conv2d(ch, max(self.student_final_dim, teacher_feature_dim), 1, 1, 0),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                ))
        
        # Per-stage CFM (linear): [student_feat || teacher_feat] -> k * num_classes
        combined_dim = max(self.student_final_dim, teacher_feature_dim) + teacher_feature_dim
        self.cfm = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, teacher_feature_dim),
                nn.ReLU(),
                nn.Linear(teacher_feature_dim, k * num_classes),
            )
            for _ in student_channels
        ])
        
        # Initialize weights for projectors and CFM only
        self._init_weights()
        
        # Loss functions (official uses label smoothing=0.1)
        self.ce_loss = CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Metrics
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
    
    def _make_sep_conv(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create SepConv block matching official implementation."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )
    
    def _init_weights(self):
        """Initialize weights for NEW modules only (not student/teacher)."""
        # Only initialize stage_projectors and cfm — NOT the student or teacher
        for module_list in [self.stage_projectors, self.cfm]:
            for m in module_list.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through student model."""
        return self.student(x)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler matching official."""
        params = (
            list(self.student.parameters()) +
            list(self.stage_projectors.parameters()) +
            list(self.cfm.parameters())
        )
        
        # Official uses SGD with momentum
        optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=2e-3
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=300,  # Official uses 300 epochs
            eta_min=1e-3
        )
        
        return [optimizer], [scheduler]
    
    def _student_forward_with_features(self, x: torch.Tensor):
        """
        Run student forward pass, capturing REAL intermediate features at each stage.
        
        Unlike the old approach that faked multi-scale features by pooling the final
        feature map, this extracts actual features from different network depths,
        matching the official HCD implementation.
        
        Returns:
            stage_features: List of feature tensors from each student stage
            logits: Student output logits
        """
        features = []
        
        if hasattr(self.student, 'blocks'):
            # ---- timm EfficientNet / MobileNetV2 ----
            # MobileNetV2 stage boundaries (flattened block indices):
            #   Stage 1: idx 2  -> (24,  H/4,  W/4)
            #   Stage 2: idx 5  -> (32,  H/8,  W/8)
            #   Stage 3: idx 12 -> (96,  H/16, W/16)
            #   Stage 4: after conv_head+bn2 -> (1280, H/32, W/32)
            target_indices = {2, 5, 12}
            
            x = self.student.conv_stem(x)
            x = self.student.bn1(x)
            if hasattr(self.student, 'act1'):
                x = self.student.act1(x)
            
            block_idx = 0
            for group in self.student.blocks:
                for block in group:
                    x = block(x)
                    if block_idx in target_indices:
                        features.append(x)
                    block_idx += 1
            
            # Stage 4: after conv_head + bn2 (matching official)
            x = self.student.conv_head(x)
            x = self.student.bn2(x)
            features.append(x)
            
            # Apply remaining activation before head (if exists)
            if hasattr(self.student, 'act2'):
                x = self.student.act2(x)
            
            # Get logits through model's own head
            logits = self.student.forward_head(x)
            
            return features, logits
        
        elif hasattr(self.student, 'layer1'):
            # ---- timm ResNet ----
            x = self.student.conv1(x)
            x = self.student.bn1(x)
            x = self.student.act1(x)
            if hasattr(self.student, 'maxpool'):
                x = self.student.maxpool(x)
            
            x = self.student.layer1(x)
            features.append(x)
            x = self.student.layer2(x)
            features.append(x)
            x = self.student.layer3(x)
            features.append(x)
            x = self.student.layer4(x)
            features.append(x)
            
            logits = self.student.forward_head(x)
            return features, logits
        
        else:
            raise NotImplementedError(
                f"Cannot extract intermediate features from {type(self.student).__name__}. "
                f"Supported: EfficientNet/MobileNetV2 (timm), ResNet."
            )
    
    def extract_teacher_features(self, x: torch.Tensor):
        """
        Extract penultimate features and logits from teacher.
        Returns features as 1D vector (B, D) and logits (B, C).
        """
        with torch.no_grad():
            if hasattr(self.teacher, 'forward_features'):
                features = self.teacher.forward_features(x)
                
                if len(features.shape) == 4:  # CNN
                    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
                elif len(features.shape) == 3:  # ViT
                    features = features[:, 0]
                
                if hasattr(self.teacher, 'head'):
                    logits = self.teacher.head(features)
                elif hasattr(self.teacher, 'classifier'):
                    logits = self.teacher.classifier(features)
                elif hasattr(self.teacher, 'fc'):
                    logits = self.teacher.fc(features)
                else:
                    logits = self.teacher(x)
            elif hasattr(self.teacher, 'features') and hasattr(self.teacher, 'classifier'):
                # torchvision DenseNet
                features = self.teacher.features(x)
                features = F.relu(features, inplace=False)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                logits = self.teacher.classifier(features)
            else:
                logits = self.teacher(x)
                features = logits
        
        return features, logits
    
    def compute_hcd_loss(
        self,
        student_logits: torch.Tensor,
        student_stage_features: List[torch.Tensor],
        teacher_features: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Compute HCD loss matching the official implementation.
        
        loss_total = loss_gt + loss_kd + loss_hcd + loss_orthogonality
        
        Matches official HCD (https://github.com/yema-web/HCD):
        - loss_kd: standard KL(student || teacher) — direct distillation signal
        - loss_hcd: KL(student || sub-logits) — complementary distillation
        - loss_gt: CE(student, labels) + CE(sub-logits, labels)
        - loss_orthogonality: diversity constraint on sub-logits
        - Sub-logits are NOT detached (official trains CFM via both CE and KL)
        - Losses are SUMMED across stages (not averaged) — official behavior
        - Temperature used as-is from config (official default: 1.0)
        """
        log_dict = {}
        B = student_logits.size(0)
        
        T = self.temperature
        
        hcd_losses = []
        entropy_losses = []
        orthogonality_losses = []
        
        # Process each stage
        for stage_idx, (stage_feat, projector, cfm) in enumerate(
            zip(student_stage_features, self.stage_projectors, self.cfm)
        ):
            # 1) Project student stage features to vector
            feat_student_final = projector(stage_feat)  # (B, D)
            
            # 2) Concatenate with teacher features
            feat_cat = torch.cat([feat_student_final, teacher_features], dim=1)
            
            # 3) CFM: produce k * num_classes logits
            logits_student_head = cfm(feat_cat)  # (B, k * C)
            logits_student_head = logits_student_head.view(B, self.k, self.num_classes)
            
            # 4) Fuse with teacher logits (SDD rectification, Eq. 6)
            logits_student_head = (
                self.lambda_student * logits_student_head + 
                self.lambda_teacher * teacher_logits.unsqueeze(1).expand(-1, self.k, -1)
            )
            
            # 5) Orthogonality loss on masked sub-logits
            masked_logits = _hcd_remove_one_hot(logits_student_head, labels)
            orthogonality_losses.append(_hcd_orthogonality_loss(masked_logits, self.ortho_threshold))
            
            # 6) CE on sub-logits (trains CFM to produce good predictions)
            cross_entropy = torch.tensor(0.0, device=student_logits.device)
            for i in range(self.k):
                cross_entropy = cross_entropy + self.ce_loss(logits_student_head[:, i, :], labels)
            cross_entropy = cross_entropy / self.k
            entropy_losses.append(cross_entropy)
            
            # 7) HCD KL loss: student learns from sub-logits (NOT detached — official)
            # In the official code, gradients flow through both student AND CFM.
            local_kl = torch.tensor(0.0, device=student_logits.device)
            for i in range(self.k):
                local_kl = local_kl + F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(logits_student_head[:, i, :] / T, dim=1),
                    reduction='sum'
                ) * (T ** 2) / B
            local_kl = local_kl / self.k
            hcd_losses.append(local_kl)
        
        # SUM across stages (official behavior — not averaged)
        loss_hcd = self.hcd_loss_weight * sum(hcd_losses)
        loss_gt = self.gt_loss_weight * self.ce_loss(student_logits, labels) + \
                  self.gt_loss_weight * sum(entropy_losses)
        loss_orthogonality = self.diversity * sum(orthogonality_losses)
        
        # Standard KD loss: direct KL(student || teacher) — MISSING before this fix
        log_pred_student = F.log_softmax(student_logits / T, dim=1)
        pred_teacher = F.softmax(teacher_logits / T, dim=1)
        loss_kd = self.kd_loss_weight * F.kl_div(
            log_pred_student, pred_teacher, reduction='batchmean'
        ) * (T ** 2)
        
        loss_total = loss_gt + loss_kd + loss_hcd + loss_orthogonality
        
        # Logging
        log_dict['loss_gt'] = loss_gt
        log_dict['loss_kd'] = loss_kd
        log_dict['loss_hcd'] = loss_hcd
        log_dict['loss_orthogonality'] = loss_orthogonality
        log_dict['loss_total'] = loss_total
        
        return loss_total, log_dict
    
    def training_step(self, batch, batch_idx):
        """Training step matching official HCD — single forward pass."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        teacher_images = batch['teacher_input_0'].to(self.device)
        
        # Single student forward: extract REAL intermediate features + logits in ONE pass
        student_stage_features, student_logits = self._student_forward_with_features(student_images)
        
        # Teacher features/logits (frozen)
        teacher_features, teacher_logits = self.extract_teacher_features(teacher_images)
        
        # HCD loss (now includes loss_kd, matching official)
        loss_total, log_dict = self.compute_hcd_loss(
            student_logits=student_logits,
            student_stage_features=student_stage_features,
            teacher_features=teacher_features,
            teacher_logits=teacher_logits,
            labels=labels
        )
        
        self.log_dict({
            'train/loss_gt': log_dict['loss_gt'],
            'train/loss_kd': log_dict['loss_kd'],
            'train/loss_hcd': log_dict['loss_hcd'],
            'train/loss_orth': log_dict['loss_orthogonality'],
            'train/loss_total': log_dict['loss_total'],
        }, on_epoch=True, prog_bar=True)
        
        return loss_total
    
    def shared_eval_step(self, batch, stage: str):
        """Shared evaluation step for validation and test."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        loss = self.ce_loss(student_logits, labels)
        
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


###############################################################################
# OFA-KD: One-for-All Knowledge Distillation (NeurIPS 2023)
# Reference: https://proceedings.neurips.cc/paper_files/paper/2023/file/
#            fb8e5f198c7a5dcd48860354e38c0edc-Paper-Conference.pdf
# Official implementation: https://github.com/Hao840/OFAKD
###############################################################################


def _ofa_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target_mask: torch.Tensor,
    eps: float,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    OFA adaptive target enhancement loss — numerically stable version.
    
    Uses log_softmax for student (avoids log(0) in fp16) and computes
    teacher softmax in float32 to prevent exact zeros before exponentiation.
    
    Args:
        logits_student: Student logits (B, C)
        logits_teacher: Teacher logits (B, C)
        target_mask: One-hot ground truth (B, C)
        eps: Enhancement exponent (1.0 for CIFAR-100, 1.5 for ImageNet)
        temperature: Softmax temperature
        
    Returns:
        Scalar loss
    """
    # Student: use log_softmax for numerical stability (avoids log(0))
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    
    # Teacher: compute in float32 to avoid exact zeros in fp16
    logits_t_f32 = logits_teacher.float() / temperature
    pred_teacher = F.softmax(logits_t_f32, dim=1)
    
    # Target enhancement (Eq. in paper)
    target_mask_f32 = target_mask.float()
    prod = (pred_teacher + target_mask_f32) ** eps
    weights = (prod - target_mask_f32).to(log_pred_student.dtype)
    
    loss = torch.sum(-weights * log_pred_student, dim=-1)
    return loss.mean()


class _SepConv(nn.Module):
    """Separable convolution block matching official OFA-KD implementation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                      padding=1, groups=in_channels, bias=False),
            # Pointwise
            nn.Conv2d(in_channels, in_channels, kernel_size=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                      padding=1, groups=in_channels, bias=False),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        return self.layers(x)


class OFAKDLitModule(pl.LightningModule):
    """
    OFA-KD: One-for-All Knowledge Distillation (NeurIPS 2023).
    
    Bridges the gap between heterogeneous architectures by:
    1. Projecting intermediate features into an aligned latent space
    2. Using adaptive target enhancement (ATE) for logit distillation
    
    Loss: L_total = gt_loss + kd_loss + ofa_loss
    
    Reference: https://github.com/Hao840/OFAKD
    """
    
    def __init__(
        self,
        teacher_models: List[nn.Module],
        student_model: nn.Module,
        temperature: float = 4.0,
        learning_rate: float = 0.05,
        num_classes: int = 100,
        ofa_eps: float = 1.0,               # 1.0 for CIFAR-100, 1.5 for ImageNet
        ofa_loss_weight: float = 1.0,
        gt_loss_weight: float = 1.0,
        kd_loss_weight: float = 1.0,
        ofa_temperature: float = 1.0,
        label_smoothing: float = 0.1,
        student_channels: List[int] = None,
        student_final_dim: int = 1280,
        teacher_feature_dim: int = 768,
        warmup_epochs: int = 3,
        max_epochs: int = 300,
        teacher_input_size: int = None,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=["teacher_models", "student_model"])
        
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.ofa_eps = ofa_eps
        self.ofa_loss_weight = ofa_loss_weight
        self.gt_loss_weight = gt_loss_weight
        self.kd_loss_weight = kd_loss_weight
        self.ofa_temperature = ofa_temperature
        self.label_smoothing = label_smoothing
        self.teacher_feature_dim = teacher_feature_dim
        self.student_final_dim = student_final_dim
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.teacher_input_size = teacher_input_size
        
        if student_channels is None:
            student_channels = [24, 32, 96, 1280]
        self.student_channels = student_channels
        
        # Student model
        if student_model is None:
            raise ValueError("A student model must be provided.")
        self.student = student_model
        
        # Teacher model (frozen)
        if not teacher_models or len(teacher_models) == 0:
            raise ValueError("Teacher model must be provided for OFA-KD.")
        self.teacher = teacher_models[0]
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Per-stage projectors (matching official SepConv-based design)
        self.projectors = nn.ModuleList()
        for i, ch in enumerate(student_channels):
            if i < len(student_channels) - 1:
                down_sample_num = len(student_channels) - 1 - i
                layers = []
                in_ch = ch
                for j in range(down_sample_num):
                    if j == down_sample_num - 1:
                        out_ch = max(self.student_final_dim, teacher_feature_dim)
                    else:
                        out_ch = in_ch * 2
                    layers.append(_SepConv(in_ch, out_ch))
                    in_ch = out_ch
                layers.append(nn.AdaptiveAvgPool2d(1))
                layers.append(nn.Flatten())
                self.projectors.append(nn.Sequential(*layers))
            else:
                self.projectors.append(nn.Sequential(
                    nn.Conv2d(ch, max(self.student_final_dim, teacher_feature_dim), 1, 1, 0),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                ))
        
        # Per-stage linear heads: projected_dim -> num_classes
        proj_out_dim = max(self.student_final_dim, teacher_feature_dim)
        self.ofa_heads = nn.ModuleList([
            nn.Linear(proj_out_dim, num_classes)
            for _ in student_channels
        ])
        
        # Weight initialization matching official OFA-KD (distillers/utils.py:init_weights)
        self._init_ofa_weights()
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Metrics
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes, average='macro')
    
    def _init_ofa_weights(self):
        """Initialize projector and head weights matching official OFA-KD.
        
        Official uses (distillers/utils.py:init_weights):
        - Conv2d: kaiming_normal_, mode='fan_out', nonlinearity='relu'
        - BatchNorm2d: weight=1, bias=0
        - Linear: trunc_normal_(std=0.02), bias=0
        """
        for module in [self.projectors, self.ofa_heads]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.student(x)
    
    def configure_optimizers(self):
        """SGD + linear warmup + cosine annealing (matching official)."""
        params = (
            list(self.student.parameters()) +
            list(self.projectors.parameters()) +
            list(self.ofa_heads.parameters())
        )
        
        optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=2e-3
        )
        
        # Linear warmup from ~0 to lr over warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4 / max(self.learning_rate, 1e-8),
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        
        # Cosine annealing for the rest
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=1e-3,
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )
        
        return [optimizer], [scheduler]
    
    def _student_forward_with_features(self, x: torch.Tensor):
        """
        Run student forward pass, capturing REAL intermediate features at each stage.
        
        Returns:
            stage_features: List of feature tensors from each student stage
            logits: Student output logits
        """
        features = []
        
        if hasattr(self.student, 'blocks'):
            # timm EfficientNet / MobileNetV2
            target_indices = {2, 5, 12}
            
            x = self.student.conv_stem(x)
            x = self.student.bn1(x)
            if hasattr(self.student, 'act1'):
                x = self.student.act1(x)
            
            block_idx = 0
            for group in self.student.blocks:
                for block in group:
                    x = block(x)
                    if block_idx in target_indices:
                        features.append(x)
                    block_idx += 1
            
            x = self.student.conv_head(x)
            x = self.student.bn2(x)
            features.append(x)
            
            if hasattr(self.student, 'act2'):
                x = self.student.act2(x)
            
            logits = self.student.forward_head(x)
            return features, logits
        
        elif hasattr(self.student, 'layer1'):
            # timm ResNet
            x = self.student.conv1(x)
            x = self.student.bn1(x)
            x = self.student.act1(x)
            if hasattr(self.student, 'maxpool'):
                x = self.student.maxpool(x)
            
            x = self.student.layer1(x)
            features.append(x)
            x = self.student.layer2(x)
            features.append(x)
            x = self.student.layer3(x)
            features.append(x)
            x = self.student.layer4(x)
            features.append(x)
            
            logits = self.student.forward_head(x)
            return features, logits
        
        else:
            raise NotImplementedError(
                f"Cannot extract intermediate features from {type(self.student).__name__}. "
                f"Supported: EfficientNet/MobileNetV2 (timm), ResNet."
            )
    
    def extract_teacher_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Extract teacher logits using the teacher's own forward method."""
        with torch.no_grad():
            return self.teacher(x)
    
    def training_step(self, batch, batch_idx):
        """Training step matching official OFA-KD implementation.
        
        Key difference from standard KD: the KD loss also uses ofa_loss (ATE)
        with ofa_temperature, NOT standard KL divergence. This is the core of
        the OFA method — all distillation uses adaptive target enhancement.
        
        CRITICAL: In official OFA, teacher and student receive the EXACT SAME
        image.  We use student_images for both so augmentation, resolution, and
        normalisation are identical (the teacher_input may differ in crop,
        flip, size, and stats).
        """
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        
        # Single student forward: features + logits
        student_stage_features, student_logits = self._student_forward_with_features(student_images)
        
        # Teacher logits (frozen) — use student_images so teacher sees the
        # exact same augmented image (matching official OFA implementation).
        # CIFAR CNN teachers (DenseNet/ResNet) expect 32×32 input; feeding
        # them 224×224 causes OOM.  Resize to teacher_input_size when set.
        if self.teacher_input_size is not None and student_images.shape[-1] != self.teacher_input_size:
            teacher_images = F.interpolate(
                student_images,
                size=(self.teacher_input_size, self.teacher_input_size),
                mode='bilinear',
                align_corners=False,
            )
        else:
            teacher_images = student_images
        teacher_logits = self.extract_teacher_logits(teacher_images)
        
        # Ground truth loss (with label smoothing)
        loss_gt = self.gt_loss_weight * self.ce_loss(student_logits, labels)
        
        # Target mask for ATE
        target_mask = F.one_hot(labels, self.num_classes).float()
        
        # KD loss: official OFA uses ofa_loss (ATE) for final logits too,
        # NOT standard KL divergence. Uses ofa_temperature (1.0), not a
        # separate KD temperature. This is the key insight of OFA.
        loss_kd = self.kd_loss_weight * _ofa_loss(
            student_logits, teacher_logits, target_mask,
            self.ofa_eps, self.ofa_temperature
        )
        
        # OFA loss: per-stage projected features -> logits -> ATE loss
        loss_ofa = torch.tensor(0.0, device=self.device)
        
        for stage_feat, projector, head in zip(
            student_stage_features, self.projectors, self.ofa_heads
        ):
            proj_feat = projector(stage_feat)
            stage_logits = head(proj_feat)
            loss_ofa = loss_ofa + _ofa_loss(
                stage_logits, teacher_logits, target_mask,
                self.ofa_eps, self.ofa_temperature
            )
        
        loss_ofa = self.ofa_loss_weight * loss_ofa
        
        loss_total = loss_gt + loss_kd + loss_ofa
        
        self.log_dict({
            'train/loss_gt_step': loss_gt,
            'train/loss_kd_step': loss_kd,
            'train/loss_ofa_step': loss_ofa,
            'train/loss_total_step': loss_total,
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss_total
    
    def shared_eval_step(self, batch, stage: str):
        """Shared evaluation step."""
        labels = batch['label'].to(self.device)
        student_images = batch['student_input'].to(self.device)
        student_logits = self.student(student_images)
        
        loss = self.ce_loss(student_logits, labels)
        
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
        return self.shared_eval_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
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
