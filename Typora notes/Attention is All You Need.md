# CNN学习

## 归一化方法

<img src="Attention is All You Need.assets/Normolizaiton.png" alt="Normolizaiton" style="zoom:67%;" />

### Batch Norm

在图像任务中，对于输入为N\*C\*H\*W的数据，考虑到输出的每一个Channel内的数据均是通过同一个卷积核得到的，等价于对于一组N\*H\*W的数据，它们是通过同一个全连接网络得到的输出，因此这里在进行Batch_Norm的时候，需要对N\*H\*W个数据点计算mean和standard deviation (std)

### Layer Norm

对一个Batch中的单个输入的每一个数据点计算整体的mean和std，并进行归一化，因此不会涉及Batch的概念

### Instance Norm

对一个Batch中的单个输入的每一层进行归一化，不会涉及Batch的概念

### Group Norm

对一个Batch中的单个输入的若干层进行归一化，不会涉及Batch的概念

## Attention is All You Need[1706.03762]

笔记模板：https://nlp.seas.harvard.edu/2018/04/03/attention.html

传统方法的问题：为了减少序列转换时的计算，之前提出过例如ByteNet和ConvS2S等采用卷积神经网络并行提取输入和输出序列中的隐含表征信息，但在这些方法中，*关联来自两个任意输入或输出位置的信号所需的操作数量随着位置之间的距离而增长*，这使得学习到距离较远的两个位置的关联信息比较困难。

self-attention：一种注意力机制，将一个单一序列的不同位置处关联起来，以此得到该序列相应的表征。

Transformer：第一个完全依赖自注意力来计算其输入和输出表示而不使用序列对齐RNN或卷积的转换模型。

Tips：为了保证最后一个维度不损失信息，长度仍然为dmodel，通常多头的注意力机制中的每一个head最后一个维度dv与head个数的乘积保证为dmodel，并沿最后一个维度对head进行拼接即可

### 模型整体框架

<img src="Attention is All You Need.assets/Transformer_1.png" alt="Transformer_1" style="zoom:67%;" />

* 模型结构
  * Encoder
    * Position Embedding
    * Multi-head Self-attention
    * LayerNorm & Residual
    * Feedforward Neural Network
      * Linear1(large)
      * Linear2(d_model)
    * LayerNorm & Residual
  * Decoder
    * Position Embedding
    * Casual Multi-head Self-attention
    * LayerNorm & Residual
    * Memory-base Multi-head Cross-attention
    * LayerNorm & Residual
    * Feedforward Neural Network
      * Linear1(large)
      * Linear2(d_model)
    * LayerNorm & Residual

* 使用类型

  * Encoder only

    BERT、分类任务、非流式任务

  * Decoder only

    GPT系列、语言建模、自回归生成任务、流式任务

  * Encoder-Decoder

    机器翻译、语音识别

* 特点
  * 无先验假设（例如：局部关联性->CNN、有序建模型->RNN）
  * 核心计算在于自注意力机制，平方复杂度
  * 数据量的要求与先验假设的程度成反比（即如果在Transformer中加入先验假设，可以减少所需的数据量）

### 代码框架(只保留核心代码)

#### 1、Transformer(Module)类

这是一个继承自Module类的子类，Tranformer的建立通过实例化这个类来实现，主要是通过传入的参数实例化Encoder和Decoder的Layer,同时实例化一个norm Layer，并作为参数得到多层堆叠后的Encoder和Decoder

注意这里实例化的norm layer是将二者在最后一层的输出上加入的，并没有展示在上面的结构图中。而它们每一层中的LayerNorm是在堆叠的每一层layer中被实例化的。

:star::star:**:star:关于layer norm，这里所采用的LayerNorm都是对最后一个维度d_model即特征维度进行归一化，主要原因在于：**

**1、由于输入序列一般是不等长的，需要加入padding，导致尽管序列长度变为了一致，但是很多embedding是没有意义的，而有意义的embedding和它们的分布大概率是不同的，如果采用BN，可能会损失很多这部分有意义的embedding的信息**

**2、LN抹平了不同样本之间的大小关系，而保留了不同特征之间的大小关系。BN抹平了不同特征之间的大小关系，而保留了不同样本之间的大小关系，这样，如果具体任务依赖于不同样本之间的关系，BN更有效，尤其是在CV领域，例如不同图片样本进行分类，不同样本之间的大小关系得以保留。**

#### 2、TransformerEncoder(Module)/TransformerDecoder(Module)类

TransformerEncoder(Module)：这个类在初始化过程中会调用*_get_clones*函数对EncoderLayer进行复制，而在它的forward函数中，需要传入的mask参数有两个，一个是用于padding的mask，一个是用于attention的mask，在原始论文中Encoder通常不需要加入attention mask，而for循环会调用实例化好的TransformerEncoderLayer，并将每一层的输入放进去迭代。

TransformerDecoder(Module)：这个类在初始化过程中会调用*_get_clones*函数对DecoderLayer进行复制，在Decoder中，需要传入的mask参数有四个，两个是用于padding的mask，两个是用于attention的mask，在原始论文中Encoder通常不需要加入attention mask，因此这里有三个mask是必须传入的,剩下的过程与上述一样

```python
class TransformerEncoder(Module):
   
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
       "在Encoder中，需要传入的mask参数有两个，一个是用于padding的mask，一个是用于attention的mask，在原始论文中Encoder通常不需要加入attention mask"
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask,src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
 memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
 memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
         "在Decoder中，需要传入的mask参数有四个，两个是用于padding的mask，两个是用于attention的mask，在原始论文中Encoder通常不需要加入attention mask，因此这里有三个mask是必须传入的"
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
```



#### 3、TransformerEncoderLayer(Module)/TransformerDecoderLayer(Module)类

这两个类就是完成图1中每个Layer层的建立，其中注意力部分是通过实例化*MultiheadAttention*类，并传入相应的输入和mask矩阵来完成的。这里与建立一个最基本的pytorch深度学习网络的过程一致，即采用MHA、Norm、Linear几种网络进行堆叠即可。

:star::star::star:**在pytorch版本的代码实现中，在所有的MHA中，使用的head数均是相同的，唯一区别是在采用forward函数调用的过程中，需要传入不同的Q、K、V和mask矩阵**

#### 4、MultiheadAttention(Module)类

实现MHA的类，主要工作是进行一些数据维度的检查、初始化等操作。主要检查的是输入的$\bf Q$、$\bf K$、$\bf V$的维度是否符合要求，在程序中默认了*embed_dim = q_dim*，而在调用MHA函数进行计算的过程中，会拿到$\bf Q$的*embed_dim*并于传入的维度信息进行比对，如果不相等会报错。pytrorch中默认采用论文中对嵌入维的设计，在不人为给定的情况下，默认*embed_dim = q_dim = k_dim = v_dim*。输入的$\bf Q$、$\bf K$、$\bf V$会经过映射，其中每个映射矩阵的维度均为*embed_dim\*embed_dim*，同时映射的偏置的维度为*embed_dim*

在编程过程中，由于当满足默认维度的条件下，使用一个*3\*embed_dim\*embed_dim*的矩阵即可表示输入所需要的映射矩阵，因此需要通过*Parameter*新建一个映射矩阵和偏置，并注册到实例化后的*state_dict*中，表明这是模型参数，这里*Parameter*是*Tensor*的一个子类。在建立模型的过程中，当对一个类的属性进行赋值时，会调用魔法方法*\_\_setattr\_\_*，而在所有network的基类Module中对这个方法进行了重写，并利用isinstance函数检查该类是否属于*Parameter*或者*Module*

Step1:定义了一个函数，输入加上\*表明输入参数的数量不确定，并将传入的参数存储为元组，即可以是模型中不同的参数字典的集合，函数的功能是删除原始模型中名字为name的模块。

Step2:检查输入的Value是否属于*Parameter*类，如果属于，首先检查Module是否建立了实例(调用\_new\_函数建立实例会自动调用\_init_\)，如果已经初始化，就将name从原来实例中的所有字典中删去，并通过*register\_parameter*函数将name和value绑定到模型上，作为在训练中可以更新的参数。

Step3：如果不属于Parameter类，但是name在原始的模型的params中，这时候必须要求value为{}空字典，才能更新相应parameter的参数为{}

**传入类型Parameter，更新parameters，同时删除所有地方与该名字相关的内容；传入类型Module,更新modules，同时删除所有地方与该名字相关的内容；传入类型Tensor，更新Buffer**

```python
 def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):# Step1 #
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):  # Step2 # 
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:  # Step3 # 
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)
```

在建立映射参数矩阵时，采用的是*torch.empty*，这个函数返回一个没有初始化的tensor，即里面的数据完全随机，因此采用xavier_uniform_对参数进行初始化。之后需要定义forward函数，传入的$\bf Q$、$\bf K$、$\bf V$三个参数首先会将它们变成*(length, batch\_size, embed\_dim)*，并直接调用*multi_head_attention_forward*函数，得到MHA的结果，输出根据batch_first决定哪个维度在前。

```python
attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
```

#### 5、multi_head_attention_forward函数

##### （1）、检查传入的$\bf Q$、$\bf K$、$\bf V$的形状，得到src_len和tgt_len，

##### （2）、计算投影后的值

利用*_in_projection_packed*函数，

* 三个输入矩阵的维度为(length, batch\_size, embed\_dim)， 
* in_proj_weight w 的维度为(3 * embed\_dim, embed\_dim)，
* in_proj_bias b 的维度为(3 * embed\_dim)

分两种情况：

* self-attention：即$\bf Q = \bf K = \bf V$，利用线性变换函数，$y = q*w^T+b$，则$ y $的维度为(length, batch\_size, 3 * embed\_dim)，然后按最后一个维度chunk，返回一个list，每一项的维度为(length, batch\_size, embed\_dim)。这里的trick在于，由于三个矩阵是相同的，因此相当于每次都把embed_dim个特征映射成 3 * embed_dim个特征，直接切割即可。对于$b$，利用广播机制会自动补充length和batch_size两个维度，当然这是在linear函数的内部实现的问题。
* encoder-decoder attention：即仅满足$\bf K = \bf V$时，$w$矩阵首先按照split函数的默认切分维度dim = 0进行切分，得到两个投影矩阵(embed\_dim, embed\_dim)和(2 * embed\_dim, embed\_dim)，再分别做映射和分割，通过相加以Tuple的形式输出。

```python
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
```



##### （3）、准备mask矩阵->这是前期工作，主要是检查维度和数据类型

* 对于mask，只支持float byte bool三种数据类型
* attention-mask必须是一个三维矩阵，因此如果输入一个二维矩阵，先检查维度是否为(tgt_len, src_len)，如果是则扩充一个0维度，变成(1, tgt_len, src_len)，后续计算中会利用广播机制自动进行扩维。如果本身就是三维矩阵，检查维度是否为(batch\_size * heads, tgt_len, src_len)即可。
* key_padding_mask通常直接使用bool， 以输入为例，维度为（batch_size, src_len）,直接通过维度扩充到（batch_size * heads, 1, src_len）
* 最后对attention-mask和key_padding_mask使用logical_or函数或者masked_fill函数，这两个函数均会利用广播机制，将二者扩充为（batch_size * heads, tgt_len, src_len）的矩阵并进行相应的操作，最终将需要mask的部分都设置为*-inf*

:star::对于key的padding_mask，这里为什么可以通过扩维只mask掉最后padding长度的列，是因为这样可以保证每一个query此时都不会与被padding掉的位置做attention，因为这里的attention_score在这些地方都为0，而对于这些被padding掉的位置本身作为query并没有意义，就算到了下一层，attention的时候仍然会被忽略。不难想象，这里在做encoder-decoder attention的时候，query矩阵的length会变，但是在attenrion矩阵中mask还是只与key的有效长度有关，因此在程序中的padding矩阵只需要令attention矩阵最后两列为-inf，因此上述padding的信息只需要一个（batch_size, src_len）的矩阵进行存储，在实际运算中根据tgt_len进行扩维即可。

<img src="Attention is All You Need.assets/v2-6e62ae19572d34c45e98d8e0092924a6_720w.jpg" alt="v2-6e62ae19572d34c45e98d8e0092924a6_720w" style="zoom:67%;" />

<img src="Attention is All You Need.assets/v2-ee68b3009689ca74cc921237384d1993_720w.jpg" alt="v2-ee68b3009689ca74cc921237384d1993_720w" style="zoom:67%;" />

:star::对于atten_mask，主要是为了考虑时间上的顺序，掩盖掉当前时刻之后的信息，使得某一时刻tgt只能得到有限的信息，这里一般只有在self-atten的时候才需要掩盖。在程序中，可以使得一个batch的atten_mask全部为自定义，此时需要输入一个三维张量，而如果输入的是二维张量，默认每一个样本的atten是相同的，并自动扩充batch的维度。

<img src="Attention is All You Need.assets/v2-ef714b246dfc08c912db18aaec0542cb_720w.jpg" alt="v2-ef714b246dfc08c912db18aaec0542cb_720w" style="zoom:67%;" />

##### （4）、改变$\bf Q = \bf K = \bf V$的形状，并且batch_size需要放在最前面

* $\bf Q$->(tgt_len, batch\_size, embed\_dim)->(tgt_len, batch\_size*heads, head\_dim)->(batch\_size\*heads, tgt_len, head\_dim)
* $\bf K$->(src_len, batch\_size, embed\_dim)->(src_len, batch\_size*heads, head\_dim)->(batch\_size\*heads, src_len, head\_dim)
* $\bf V$->(src_len, batch\_size, embed\_dim)->(src_len, batch\_size*heads, head\_dim)->(batch\_size\*heads, src_len, head\_dim)

注意这里变换使用了*transpose*，张量存储过程中不再满足连续性条件。

##### （5）、将$\bf Q = \bf K = \bf V$送入_scaled_dot_product_attention函数，同时送入mask和dropout参数（注意这里dropout\_p在train=False时会被置零）

在做矩阵乘法时，调用*torch.bmm*函数，接收三维输入，b * n * m，最后两维进行矩阵相乘，得到atten矩阵，维度为(batch\_size\*heads, tgt_len, src_len)，加入dropout，并与$\bf V$相乘，得到的输出为(batch\_size\*heads, tgt_len, head\_dim)，这里等价于得到了batch\_size\*heads个特征为head_dim的特征矩阵

##### （6）、调整矩阵维度进行输出

为了能进行维度的整合，首先需要让张量的存储空间连续。因此将上述输出的0、1维度转置后，调用contiguous函数，当保证存储连续后，调用view函数，合并成embed_dim个特征维度的张量，因此输出的维度为(tgt_len, batch\_size, embed\_dim)，然后根据out_proj_weight和out_proj_bias，经过一层Linear层映射得到最终输出
