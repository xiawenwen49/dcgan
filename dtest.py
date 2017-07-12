# from tkinter import *
# import tkinter.messagebox as messagebox
#
#
# def ui_process():
#     root = Tk()
#     root.geometry("600x400")
#
#
#     L_load=Label(root, text='load status')
#     L_load.config(font='Helvetica -16')
#     L_load.place(x=60, y=15, anchor="w")
#
#     E_load=Entry(root)
#     E_load.config(font='Helvetica -16')
#     E_load.place(x=60, y=30)
#
#     B_run=Button(root, text="run", command=hello)
#     B_run.config(font='Helvetica -18 ')
#     B_run.place(x=60, y=70)
#
#     root.mainloop()
#
# def hello():
#     print("helloword")
#
# if __name__=="__main__":
#
#     ui_process()

import mxnet as mx
user = mx.symbol.Variable('user')
item = mx.symbol.Variable('item')
score = mx.symbol.Variable('score')

# Set dummy dimensions
k = 64
max_user = 100
max_item = 50

# user feature lookup
user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)

# item feature lookup
item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)

# predict by the inner product, which is elementwise product and then sum
net = user * item
net = mx.symbol.sum_axis(data = net, axis = 1)
net = mx.symbol.Flatten(data = net)

# loss layer
net = mx.symbol.LinearRegressionOutput(data = net, label = score)

# Visualize your network
mx.viz.plot_network(net)


