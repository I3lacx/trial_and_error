# Tensorflow Guide

This is a very basic collection of usefull commands and things I tend to forget

### Tensorboard

With tensorboard you can display your current mess of connections in
variables with some simple commands.
Inside your Code use:

```python
import tensorflow as tf
#... add some Code ...
sess = tf.Session()
#... add some Code ..
file_writer = tf.summary.FileWriter('path/to/logs/', sess.graph)
```

Inside your current folder open cmd and type:

			tensorboard --logdir path/to/logs/

This will open a localhost website on the given Port. You can then connect to: http://localhost:6006 to view the website.
