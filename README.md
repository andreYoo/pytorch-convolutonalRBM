# pytorch-Convolutional RBM
Single layer Convolutional RBM based on Pytorch Lib </br>
(Original RBM is referred from https://github.com/GabrielBianconi/pytorch-rbm) </br>
(Structural details is referred from "Convolutional Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical Representations" of Lee et al.) </br>



*This model consists of single CovRBM layers, and it is implemented roughly, so there are GPU memory management issues on this source code. We will address these issues sooner or later.

File configures
.</br>
├── convrbm_example.py </br>
├── convrbm.py </br>
├── data </br>
│   └── mnist </br>
│       └── MNIST </br>
│           ├── processed </br>
│           │   ├── test.pt </br>
│           │   └── training.pt </br>
│           └── raw </br>
│               ├── t10k-images-idx3-ubyte </br>
│               ├── t10k-images-idx3-ubyte.gz </br>
│               ├── t10k-labels-idx1-ubyte </br>
│               ├── t10k-labels-idx1-ubyte.gz </br>
│               ├── train-images-idx3-ubyte </br>
│               ├── train-images-idx3-ubyte.gz </br> 
│               ├── train-labels-idx1-ubyte </br>
│               └── train-labels-idx1-ubyte.gz </br>
├── LICENSE </br>
├── __pycache__ </br>
│   ├── convrbm.cpython-36.pyc </br>
│   └── utils.cpython-36.pyc </br>
├── README.md </br>
└── utils.py </br>


How to run</br>
python convrbm_example.py</br>

Running Output</br>

Loading dataset...</br>
Training ConvRBM...</br>
Epoch Error (epoch=0): 6593016.5000</br>
Epoch Error (epoch=1): :3434561.7500</br>
Epoch Error (epoch=2): 2627748.5000</br>
Epoch Error (epoch=3): 2355443.5000</br>
Epoch Error (epoch=9): 1962103.7500</br>




dependendies. 
Python 3.6
