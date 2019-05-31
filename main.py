from train_routines import xentropy
import os

xentropy.train(os.getcwd() + "/runs/", 32, 1, 0.05)