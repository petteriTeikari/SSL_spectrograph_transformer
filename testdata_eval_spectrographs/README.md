# Data formatting

You need to have the `torchvision` formatting here (so you don't have to modify the code):

e.g.

```
testdata_eval_spectrographs/train/class_1
testdata_eval_spectrographs/train/class_2
testdata_eval_spectrographs/train/class_3
testdata_eval_spectrographs/train/class_4
testdata_eval_spectrographs/train/class_5

testdata_eval_spectrographs/val/class_1
testdata_eval_spectrographs/val/class_2
testdata_eval_spectrographs/val/class_3
testdata_eval_spectrographs/val/class_4
testdata_eval_spectrographs/val/class_5
```

Now you have 5 classes as the default `eval_linear.py` uses top5 metrics so tweak the code so that you do top2 for example for two classes (healthy vs. AD), or set the k programmatically? 