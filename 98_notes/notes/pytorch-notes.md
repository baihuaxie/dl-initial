## PyTorch



----

##### nn.Conv2d

input tensor = [N, C~in~, H~in~, W~in~]

output tensor = [N, C~out~, H~out~, W~out~]

```python
# nn.Conv2d
torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')
"""
	Args:
		in_channels: (int) number of input channels
		out_channels: (int) number of output channels
		kernel_size: (int or tuple) size of convolution kernel
		stride: (int or tuple) stride of convoluton; default=1 (no striding)
		padding: (int or tuple) zero-padding added to both sides of the input; default=0 (no padding)
		padding_mode: (string; optional) 'zeros', 'reflect', 'replicate' or 'circular'; default='zeros'
		dilation: (int or tuple) spacing between kernel elements (used to increase receptive filed); default=1 (no dilation)
		groups: (int) number of blocked connections from input to output (convolution only applies to channels in the block)
		bias: (bool) if True, adds a learnable bias parameter to the ouptut; default=True
"""
```

$$
\begin{split}
H_{out} &= \frac{H_{in}+2*padding[0]-dilation[0]*(kernel\_size[0]-1)-1}{stride[0]}+1\\
W_{out} &= \frac{W_{in}+2*padding[1]-dilation[1]*(kernel\_size[1]-1)-1}{stride[1]}+1
\end{split}
$$

* if groups = in_channels: becomes depth-wise seperable convolution
* if dilation=1, padding=1, kernel_size=3, stride=1, then H~out~ = H~in~ 
  * hence for the commonly used conv3x3 kernel, it is customary to set dilation=padding=stride=1 to avoid dimension mismatch
  * this also works if stride=2 (in this case H~out~ = 1/2 * H~in~; the fraction in the formula could be rounded down)
* if kernel_size = 1, padding=0, dilation=arbitray, stride=1, then H~out~ = H~in~ 

----

##### torch.optim.lr_scheduler

| Name                                                         | Schedule                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ```LambdaLR(optimizer, lr_lambda)```                         | sets lr = initial_lr * lr_lambda (function)                  |
| ```MultiplicativeLR(optimizer, lr_lambda)```                 | sets lr = initial_lr * factor (computed by the lambda function) |
| ```StepLR(optimizer, step_size, gamma)```                    | decays lr by multiplying with the gamma factor every step_size epochs |
| ```MultiStepLR(optimizer, milestones, gamma)```              | decays lr by multiplying with the gamma factor when epochs reach each milestone |
| ```ExponentialLR(optimizer, gamma)```                        | decays lr by gamma every epoch                               |
| ```CosineAnnealingLR(optimizer, Tmax, eta_min)```            | $$\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})(1+cos(\frac{T_{cur}}{T_{max}}\pi))\hspace{10pt}\eta_{max}=\text{initial lr}\hspace{10pt}T_{cur}=\text{epochs}$$ |
| ```CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min)``` | $$\begin{split}\eta_t&=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})(1+cos(\frac{T_{cur}}{T_{i}}\pi))\\\eta_{max}&=\text{initial lr}\\T_{cur}&=\text{epochs since last restart}\\T_i&=\text{epochs between two restarts}\end{split}$$ |
| ```ReduceLROnPlateau(optimizer, mode, factor, patience, threshold, cooldown, min_lr, eps)``` | reduce learning rate by **factor** after a monitored metric has stopped improving for a **patience** # of epochs<br />usage: ```scheduler.step(metric)``` |
| ```CyclicLR(optimizer, base_lr, max_lr, mode, step_size_up, step_size_down, ...)``` | Using cyclic learning rate scheduling for each batch in three modes: triangular, triangular2, exp_range<br />usage: call ```scheduler.step()``` every **batch** |
| ```OneCycleLR(optimizer, max_lr, total_steps, pct_start, anneal_strategy, div_factor, final_div_factor, ...)``` | anneals lr from initial_lr to max_lr for pct_start*total_steps by either cosine or linear annealing strategy; then reduce from max_lr to a <br />min_lr = max_lr/div_factor/final_div_factor for the rest of the training<br />usage: call ```scheduler.step()``` every **batch** |

* usually has multiple groups of learning rates, each can be applied with different scheduling + different initial learning rates

  * the schedulers in pytorch can handle multiple groups of learning rates

* most of the lr_schedulers can be chained, i.e., lr can be simultaneously altered by multiple schedulers

  * exception is OneCycleLR

* lr_schedulers are applied either 

  * for several epochs or for each epoch after train() or after validation()

  * within the train() loop for several batches or for each batch

    