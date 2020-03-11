---

## What has been done

To add q-fair algorithms proposed in [Fair Resource Allocation](https://openreview.net/forum?id=ByexElSYDr), I (Nova-xiao) modified codes in LEAF-dev.

## Modifying idea

* I mainly modified **server.py** under the folder *models*. Because the function update_model() distributes aggregation algorithm, I added algotithm 'qfFedAvg' here. Considering the support of minibatch-SGD from LEAF and that qfFedSGD is just a special case of qfFedAvg (which is my opinion...maybe not completely right), I just added one algorithm instead of two.
* My modifying method is to transfer code in **qffedavg.py** in q-FFL repository to server.py. But There are some differences between LEAF and the simulating system used by q-FFL, so I also modified code in config.py tf_utils.py.

## Details about code changes

1. /models/utils/config.py

```python
...
        self.user_trace = False
        self.no_training = False
        self.q = -1     # factor of q-fair algorithms, -1 means not using
        
        logger.info('read config from {}'.format(config_file))
        self.read_config(config_file)
...
				elif line[0] == 'max_sample' :
             self.max_sample = int(line[1])
        elif line[0] == 'q':
             #Newly added, for supporting q-fair algorithms
             self.q = int(line[1])
...
```

2. /models/utils/tf_utils.py

```python
import numpy as np
...
# Copied from q-FFL repository
def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm

    client_grads = grad_list[0] # shape now: (784, 26)

    for i in range(1, len(grad_list)):
        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array

    return np.sum(np.square(client_grads))
```

3. /models/server.py

* This is the place where core changes are. First is adding algorithm 'qfFedAvg' to update_model(), this part is easy to find in codes.
* Second, n order to get params needed in qfFedAvg, I need to add **'before'** (old model) and **'old_loss'** (loss of old model on training dataset) to **self.updates**, which results in tuples in self.updates changing from threee elements to five elements. These changes are in train_model(), and all located in the loop where clients are traversed.
* Third, the algorithm needs the norm_grad() function which has been added to tf_utils.py before. So import norm_grad from utils.tf_utils.

## How to use new algorithm

* Change config's 'aggregate_algorithm' to 'qfFedAvg', and set 'q' to an interger and make sure q is not minus. Other things are the same with FedAvg's config.
* To use qfFedSGD, you need to set minibatch, which is a fraction between 0 and 1. Other things are the same with qfFedAvg's config.

## Test Result: Questions/Bugs

I tested qfFedAvg in comparison with FedAvg and SucFedAvg. However, due to the limit of my resources, I can only run experiment with small hyperparameters. My GPU is NVIDIA GTX 1050, and it only has 2GB's memory, so I changed 'clients_per_round' to 10,  which significantly reduced GPU memory cost. I also changed 'num_rounds' to 3000 to save time.

According to my experiment, the convergencing speeds are as follow: FedAvg < qfFedAvg(q = 5) < SucFedAvg. I don't understand why SucFedAvg onvergences faster, **maybe my config is too small to cause problem or there are some bugs in the new code**.



