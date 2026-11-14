"""Load a Saluki half-life model (Keras 2.4, Basenji codebase) under current TensorFlow.

Only one custom layer is needed. `StochasticShift` is a TRAINING-TIME augmentation that randomly shifts the
sequence by up to `shift_max` positions; at inference Basenji's implementation returns the input unchanged,
so a pass-through reproduces inference exactly. Defining it this way avoids vendoring all of Basenji.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tf_keras as keras


@keras.utils.register_keras_serializable(package="basenji")
class StochasticShift(keras.layers.Layer):
    def __init__(self, shift_max=0, symmetric=True, pad="uniform", **kw):
        super().__init__(**kw)
        self.shift_max, self.symmetric, self.pad = shift_max, symmetric, pad

    def call(self, x, training=None):
        return x                      # inference is identity; augmentation only applies during training

    def get_config(self):
        c = super().get_config()
        c.update(shift_max=self.shift_max, symmetric=self.symmetric, pad=self.pad)
        return c


def load(path):
    return keras.models.load_model(path, compile=False,
                                   custom_objects={"StochasticShift": StochasticShift})


if __name__ == "__main__":
    import sys
    m = load(sys.argv[1] if len(sys.argv) > 1
             else "datasets/deeplearning/train_gru/f0_c0/train/model0_best.h5")
    print("LOADED |", f"{m.count_params():,}", "params |", len(m.layers), "layers")
    print("input :", m.input_shape)
    print("output:", m.output_shape)
