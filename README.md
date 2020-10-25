# FW_RNN_optimizer
This is the PyTorch code associated with the proposed optimizer used in the paper RNN Training along Locally Optimal Trajectoriesvia Frank-Wolfe Algorithm, accepted by ICPR, 2020.

## Usage:

`from fgsm import FGSM, MultipleOptimizer`

To create optimizer

`optimizers = [FGSM(model.parameters(), lr=1e-3, iterT=1, mergeadam=True)] 
                        [Adam(model.parameters(), lr=self.lr)]`



`optimizer = MultipleOptimizer(optimizers)`

To update model parameter

`optimizer.step(total_batches)` where total_batches is the current batch number.

## Citation:
@article{yue2020rnn,
  title={RNN Training along Locally Optimal Trajectoriesvia Frank-Wolfe Algorithm},
  author={Yue, Yun and Li, Ming and Saligrama, Venkatesh and Zhang, Ziming},
  journal={arXiv preprint arXiv:2010.05397},
  year={2020}
}
