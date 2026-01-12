"""Model architectures for knowledge distillation"""

from .student import create_student_model
from .teacher import create_teacher_models
from .kd_module import (
    CAWeightedKDLitModule,
    DynamicKDLitModule,
    ConfidenceBasedKDLitModule,
    PATKDLitModule,
    RegionAwareAttention,
    AdaptiveFeedbackPrompt,
    BaselineStudentModule
)

__all__ = [
    "create_student_model",
    "create_teacher_models",
    "CAWeightedKDLitModule",
    "DynamicKDLitModule",
    "ConfidenceBasedKDLitModule",
    "PATKDLitModule",
    "RegionAwareAttention",
    "AdaptiveFeedbackPrompt",
    "BaselineStudentModule",
]

