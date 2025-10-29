# WandB Naming Standard

## Standardized Configuration

### Single Project for All Experiments
```yaml
project: "KD-CIFAR100"
```

**Rationale:** All experiments in one project for easy comparison

### Experiment Naming Convention

**Format:** `{method}-{architecture}-{variant}`

#### Baseline
```yaml
name: "Baseline-MobileNetV2"
```

#### Single Teacher (Method 0)
```yaml
name: "Single-{teacher}-alpha{alpha}"
# Examples:
# - Single-DenseNet121-alpha0.50
# - Single-ResNet50-alpha0.75
# - Single-ViT-alpha0.25
```

#### Method 1: CA-WKD
```yaml
name: "Method1-{teachers}"
# Example: Method1-ResNet-DenseNet-ViT
```

#### Method 2: Dynamic α-Guided
```yaml
name: "Method2-{teachers}-gamma{gamma}"
# Example: Method2-3xResNet-gamma5.0
```

#### Method 3: Adaptive α-Guided (Best)
```yaml
name: "Method3-{teachers}-temp{temp}"
# Examples:
# - Method3-2xResNet-DenseNet-temp8
# - Method3-3xResNet-NoCE-temp4
# - Method3-Diverse-temp4
```

## Updated Config Files

See below for standardized versions of all configs.

