# Keras 3.x Compatibility Fixes

The cymetric package was originally developed for older versions of TensorFlow/Keras. When using newer versions (TensorFlow 2.16+ / Keras 3.x), several compatibility issues need to be addressed. The following fixes have been applied to ensure proper functionality:

## 1. Model Attribute Compatibility (`cymetric/cymetric/models/tfmodels.py`)

**Issue**: Newer Keras versions use `_jit_compile` instead of `_is_compiled`
```python
# Fixed line ~170: Check for compilation compatibility
is_compiled = hasattr(self, '_jit_compile') and self._jit_compile
```

**Issue**: `compiled_loss` and `compiled_metrics` objects may not have `metrics` attribute
```python
# Fixed lines ~175-180: Added safety checks
if self.compiled_loss is not None and hasattr(self.compiled_loss, 'metrics'):
    metrics += self.compiled_loss.metrics
if self.compiled_metrics is not None and hasattr(self.compiled_metrics, 'metrics'):
    metrics += self.compiled_metrics.metrics
```

## 2. GradientTape Compatibility

**Issue**: `tape.watch()` on trainable variables causes "numpy() is only available when eager execution is enabled" error
```python
# Fixed line ~214: Removed redundant tape.watch() call
# trainable variables are automatically watched by GradientTape in TF 2.x
with tf.GradientTape(persistent=False) as tape:
    trainable_vars = self.model.trainable_variables
    # Removed: tape.watch(trainable_vars)  # This line was causing issues
```

## 3. Metrics Update Compatibility

**Issue**: Keras 3.x `compiled_metrics` incompatible with cymetric custom metrics
```python
# Fixed lines ~257-260: Removed compiled_metrics.update_state() calls
# The custom_metrics section properly handles cymetric-specific metrics
# Skip compiled_metrics for cymetric models to avoid compatibility issues
```

## 4. Input Layer Shape Fix (`LearningG2/models/cy_model.py`)

**Issue**: Keras 3.x requires tuple for Input layer shape parameter
```python
# Fixed: Use tuple instead of integer for shape
nn_phi.add(tf.keras.Input(shape=(n_in,)))  # Note the comma for tuple
```

## 5. TensorFlow Probability Replacement (`LearningG2/geometry/compression.py`)

**Issue**: TensorFlow Probability dependency removed from environment
```python
# Replaced tfp.math.fill_triangular functions with pure TensorFlow implementations
# Custom implementations maintain same functionality without tfp dependency
```

## Installation Instructions

After downloading and setting up the cymetric package, replace the original `tfmodels.py` file with the compatibility-fixed version in this folder:

```bash
# Navigate to your project directory
cd /path/to/your/project

# Backup the original file
cp cymetric/cymetric/models/tfmodels.py cymetric/cymetric/models/tfmodels.py.backup

# Copy the fixed version
cp LearningG2/environment/cymetric_compatibility_fixes/tfmodels.py cymetric/cymetric/models/tfmodels.py
```

## Verification

After applying these fixes, the training pipeline should work without errors

**Note**: These fixes ensure compatibility with TensorFlow 2.16+ and Keras 3.x while maintaining the original functionality of the cymetric training pipeline.
