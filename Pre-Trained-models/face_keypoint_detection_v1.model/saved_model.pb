хН
І§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02v2.1.0-0-ge5bf8de4108и

conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
: *
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
: *
dtype0

conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
: *
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
:*
dtype0

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
:*
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:*
dtype0
z
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
2*
shared_namedense_9/kernel
s
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel* 
_output_shapes
:
2*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:2*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:2
*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_18/kernel/m

+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_18/bias/m
{
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_19/kernel/m

+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_19/bias/m
{
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_20/kernel/m

+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_20/bias/m
{
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
_output_shapes
:*
dtype0

Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
2*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m* 
_output_shapes
:
2*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*'
shared_nameAdam/dense_10/kernel/m

*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:2
*
dtype0

Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:
*
dtype0

Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_18/kernel/v

+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_18/bias/v
{
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_19/kernel/v

+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_19/bias/v
{
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_20/kernel/v

+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_20/bias/v
{
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
_output_shapes
:*
dtype0

Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
2*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v* 
_output_shapes
:
2*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:2*
dtype0

Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*'
shared_nameAdam/dense_10/kernel/v

*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:2
*
dtype0

Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
э7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ј7
value7B7 B7
Ю
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
 regularization_losses
!trainable_variables
"	variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
є
0iter

1beta_1

2beta_2
	3decay
4learning_ratem]m^m_m`mamb$mc%md*me+mfvgvhvivjvkvl$vm%vn*vo+vp
 
F
0
1
2
3
4
5
$6
%7
*8
+9
F
0
1
2
3
4
5
$6
%7
*8
+9

5non_trainable_variables

6layers
	regularization_losses
7layer_regularization_losses

trainable_variables
	variables
8metrics
 
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

9non_trainable_variables

:layers
regularization_losses
;layer_regularization_losses
trainable_variables
	variables
<metrics
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

=non_trainable_variables

>layers
regularization_losses
?layer_regularization_losses
trainable_variables
	variables
@metrics
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

Anon_trainable_variables

Blayers
regularization_losses
Clayer_regularization_losses
trainable_variables
	variables
Dmetrics
 
 
 

Enon_trainable_variables

Flayers
 regularization_losses
Glayer_regularization_losses
!trainable_variables
"	variables
Hmetrics
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1

Inon_trainable_variables

Jlayers
&regularization_losses
Klayer_regularization_losses
'trainable_variables
(	variables
Lmetrics
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1

Mnon_trainable_variables

Nlayers
,regularization_losses
Olayer_regularization_losses
-trainable_variables
.	variables
Pmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6
 

Q0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	Rtotal
	Scount
T
_fn_kwargs
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

R0
S1

Ynon_trainable_variables

Zlayers
Uregularization_losses
[layer_regularization_losses
Vtrainable_variables
W	variables
\metrics

R0
S1
 
 
 
}
VARIABLE_VALUEAdam/conv2d_18/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_18/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_7Placeholder*/
_output_shapes
:џџџџџџџџџxx*
dtype0*$
shape:џџџџџџџџџxx
Ы
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7conv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_232220
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
м
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__traced_save_232511
ѓ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/v*1
Tin*
(2&*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__traced_restore_232634Н
щ
о
E__inference_conv2d_19_layer_call_and_return_conditional_losses_232003

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs

a
E__inference_flatten_6_layer_call_and_return_conditional_losses_232051

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ Т 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџxx:& "
 
_user_specified_nameinputs
Х
Ћ
*__inference_conv2d_20_layer_call_fn_232032

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_2320242
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
!
й
C__inference_model_5_layer_call_and_return_conditional_losses_232148

inputs,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2+
'dense_10_statefulpartitionedcall_args_1+
'dense_10_statefulpartitionedcall_args_2
identityЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ dense_10/StatefulPartitionedCallЂdense_9/StatefulPartitionedCallК
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_2319822#
!conv2d_18/StatefulPartitionedCallо
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_2320032#
!conv2d_19/StatefulPartitionedCallо
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_2320242#
!conv2d_20/StatefulPartitionedCallъ
flatten_6/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:џџџџџџџџџ*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2320512
flatten_6/PartitionedCallФ
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ2*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2320702!
dense_9/StatefulPartitionedCallЯ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0'dense_10_statefulpartitionedcall_args_1'dense_10_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2320922"
 dense_10/StatefulPartitionedCallЎ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ѕ
Њ
)__inference_dense_10_layer_call_fn_232376

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2320922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ш	
м
C__inference_dense_9_layer_call_and_return_conditional_losses_232352

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ш	
м
C__inference_dense_9_layer_call_and_return_conditional_losses_232070

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Х
Ъ
(__inference_model_5_layer_call_fn_232330

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2321832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Х
Ъ
(__inference_model_5_layer_call_fn_232315

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2321482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ѕ
Љ
(__inference_dense_9_layer_call_fn_232359

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ2*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2320702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ю
М
"__inference__traced_restore_232634
file_prefix%
!assignvariableop_conv2d_18_kernel%
!assignvariableop_1_conv2d_18_bias'
#assignvariableop_2_conv2d_19_kernel%
!assignvariableop_3_conv2d_19_bias'
#assignvariableop_4_conv2d_20_kernel%
!assignvariableop_5_conv2d_20_bias%
!assignvariableop_6_dense_9_kernel#
assignvariableop_7_dense_9_bias&
"assignvariableop_8_dense_10_kernel$
 assignvariableop_9_dense_10_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count/
+assignvariableop_17_adam_conv2d_18_kernel_m-
)assignvariableop_18_adam_conv2d_18_bias_m/
+assignvariableop_19_adam_conv2d_19_kernel_m-
)assignvariableop_20_adam_conv2d_19_bias_m/
+assignvariableop_21_adam_conv2d_20_kernel_m-
)assignvariableop_22_adam_conv2d_20_bias_m-
)assignvariableop_23_adam_dense_9_kernel_m+
'assignvariableop_24_adam_dense_9_bias_m.
*assignvariableop_25_adam_dense_10_kernel_m,
(assignvariableop_26_adam_dense_10_bias_m/
+assignvariableop_27_adam_conv2d_18_kernel_v-
)assignvariableop_28_adam_conv2d_18_bias_v/
+assignvariableop_29_adam_conv2d_19_kernel_v-
)assignvariableop_30_adam_conv2d_19_bias_v/
+assignvariableop_31_adam_conv2d_20_kernel_v-
)assignvariableop_32_adam_conv2d_20_bias_v-
)assignvariableop_33_adam_dense_9_kernel_v+
'assignvariableop_34_adam_dense_9_bias_v.
*assignvariableop_35_adam_dense_10_kernel_v,
(assignvariableop_36_adam_dense_10_bias_v
identity_38ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1ќ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*
valueўBћ%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesч
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Њ
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_18_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_18_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_19_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_19_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_20_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_20_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_9_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_9_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_10_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_10_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Є
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_18_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ђ
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_18_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Є
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_19_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ђ
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_19_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Є
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_20_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Ђ
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_20_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Ђ
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_9_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24 
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_9_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ѓ
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_10_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ё
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_10_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Є
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_18_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ђ
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_18_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Є
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_19_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Ђ
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_19_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Є
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_20_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ђ
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_20_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ђ
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_9_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34 
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_9_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Ѓ
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_10_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Ё
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_10_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*Ћ
_input_shapes
: :::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
Ш
Ы
(__inference_model_5_layer_call_fn_232161
input_7"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_7statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2321482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_7
Ђ
Ч
$__inference_signature_wrapper_232220
input_7"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_7statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_2319692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_7
щ
н
D__inference_dense_10_layer_call_and_return_conditional_losses_232092

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ш
Ы
(__inference_model_5_layer_call_fn_232196
input_7"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_7statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2321832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_7
3

C__inference_model_5_layer_call_and_return_conditional_losses_232300

inputs,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identityЂ conv2d_18/BiasAdd/ReadVariableOpЂconv2d_18/Conv2D/ReadVariableOpЂ conv2d_19/BiasAdd/ReadVariableOpЂconv2d_19/Conv2D/ReadVariableOpЂ conv2d_20/BiasAdd/ReadVariableOpЂconv2d_20/Conv2D/ReadVariableOpЂdense_10/BiasAdd/ReadVariableOpЂdense_10/MatMul/ReadVariableOpЂdense_9/BiasAdd/ReadVariableOpЂdense_9/MatMul/ReadVariableOpГ
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOpС
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx *
paddingSAME*
strides
2
conv2d_18/Conv2DЊ
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOpА
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx 2
conv2d_18/BiasAdd~
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx 2
conv2d_18/ReluГ
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_19/Conv2D/ReadVariableOpз
conv2d_19/Conv2DConv2Dconv2d_18/Relu:activations:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx*
paddingSAME*
strides
2
conv2d_19/Conv2DЊ
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOpА
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
conv2d_19/BiasAdd~
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
conv2d_19/ReluГ
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOpз
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx*
paddingSAME*
strides
2
conv2d_20/Conv2DЊ
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOpА
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
conv2d_20/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ Т 2
flatten_6/Const
flatten_6/ReshapeReshapeconv2d_20/Relu:activations:0flatten_6/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2
flatten_6/ReshapeЇ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMulflatten_6/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dense_9/MatMulЄ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_9/BiasAdd/ReadVariableOpЁ
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dense_9/ReluЈ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02 
dense_10/MatMul/ReadVariableOpЂ
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_10/MatMulЇ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOpЅ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_10/BiasAddР
IdentityIdentitydense_10/BiasAdd:output:0!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
щ
о
E__inference_conv2d_20_layer_call_and_return_conditional_losses_232024

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Л;
џ
!__inference__wrapped_model_231969
input_74
0model_5_conv2d_18_conv2d_readvariableop_resource5
1model_5_conv2d_18_biasadd_readvariableop_resource4
0model_5_conv2d_19_conv2d_readvariableop_resource5
1model_5_conv2d_19_biasadd_readvariableop_resource4
0model_5_conv2d_20_conv2d_readvariableop_resource5
1model_5_conv2d_20_biasadd_readvariableop_resource2
.model_5_dense_9_matmul_readvariableop_resource3
/model_5_dense_9_biasadd_readvariableop_resource3
/model_5_dense_10_matmul_readvariableop_resource4
0model_5_dense_10_biasadd_readvariableop_resource
identityЂ(model_5/conv2d_18/BiasAdd/ReadVariableOpЂ'model_5/conv2d_18/Conv2D/ReadVariableOpЂ(model_5/conv2d_19/BiasAdd/ReadVariableOpЂ'model_5/conv2d_19/Conv2D/ReadVariableOpЂ(model_5/conv2d_20/BiasAdd/ReadVariableOpЂ'model_5/conv2d_20/Conv2D/ReadVariableOpЂ'model_5/dense_10/BiasAdd/ReadVariableOpЂ&model_5/dense_10/MatMul/ReadVariableOpЂ&model_5/dense_9/BiasAdd/ReadVariableOpЂ%model_5/dense_9/MatMul/ReadVariableOpЫ
'model_5/conv2d_18/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_5/conv2d_18/Conv2D/ReadVariableOpк
model_5/conv2d_18/Conv2DConv2Dinput_7/model_5/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx *
paddingSAME*
strides
2
model_5/conv2d_18/Conv2DТ
(model_5/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_5/conv2d_18/BiasAdd/ReadVariableOpа
model_5/conv2d_18/BiasAddBiasAdd!model_5/conv2d_18/Conv2D:output:00model_5/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx 2
model_5/conv2d_18/BiasAdd
model_5/conv2d_18/ReluRelu"model_5/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx 2
model_5/conv2d_18/ReluЫ
'model_5/conv2d_19/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_5/conv2d_19/Conv2D/ReadVariableOpї
model_5/conv2d_19/Conv2DConv2D$model_5/conv2d_18/Relu:activations:0/model_5/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx*
paddingSAME*
strides
2
model_5/conv2d_19/Conv2DТ
(model_5/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_5/conv2d_19/BiasAdd/ReadVariableOpа
model_5/conv2d_19/BiasAddBiasAdd!model_5/conv2d_19/Conv2D:output:00model_5/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
model_5/conv2d_19/BiasAdd
model_5/conv2d_19/ReluRelu"model_5/conv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
model_5/conv2d_19/ReluЫ
'model_5/conv2d_20/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_5/conv2d_20/Conv2D/ReadVariableOpї
model_5/conv2d_20/Conv2DConv2D$model_5/conv2d_19/Relu:activations:0/model_5/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx*
paddingSAME*
strides
2
model_5/conv2d_20/Conv2DТ
(model_5/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_5/conv2d_20/BiasAdd/ReadVariableOpа
model_5/conv2d_20/BiasAddBiasAdd!model_5/conv2d_20/Conv2D:output:00model_5/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
model_5/conv2d_20/BiasAdd
model_5/conv2d_20/ReluRelu"model_5/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
model_5/conv2d_20/Relu
model_5/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ Т 2
model_5/flatten_6/ConstН
model_5/flatten_6/ReshapeReshape$model_5/conv2d_20/Relu:activations:0 model_5/flatten_6/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2
model_5/flatten_6/ReshapeП
%model_5/dense_9/MatMul/ReadVariableOpReadVariableOp.model_5_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02'
%model_5/dense_9/MatMul/ReadVariableOpП
model_5/dense_9/MatMulMatMul"model_5/flatten_6/Reshape:output:0-model_5/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_5/dense_9/MatMulМ
&model_5/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_5_dense_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02(
&model_5/dense_9/BiasAdd/ReadVariableOpС
model_5/dense_9/BiasAddBiasAdd model_5/dense_9/MatMul:product:0.model_5/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_5/dense_9/BiasAdd
model_5/dense_9/ReluRelu model_5/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_5/dense_9/ReluР
&model_5/dense_10/MatMul/ReadVariableOpReadVariableOp/model_5_dense_10_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02(
&model_5/dense_10/MatMul/ReadVariableOpТ
model_5/dense_10/MatMulMatMul"model_5/dense_9/Relu:activations:0.model_5/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
model_5/dense_10/MatMulП
'model_5/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'model_5/dense_10/BiasAdd/ReadVariableOpХ
model_5/dense_10/BiasAddBiasAdd!model_5/dense_10/MatMul:product:0/model_5/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
model_5/dense_10/BiasAdd
IdentityIdentity!model_5/dense_10/BiasAdd:output:0)^model_5/conv2d_18/BiasAdd/ReadVariableOp(^model_5/conv2d_18/Conv2D/ReadVariableOp)^model_5/conv2d_19/BiasAdd/ReadVariableOp(^model_5/conv2d_19/Conv2D/ReadVariableOp)^model_5/conv2d_20/BiasAdd/ReadVariableOp(^model_5/conv2d_20/Conv2D/ReadVariableOp(^model_5/dense_10/BiasAdd/ReadVariableOp'^model_5/dense_10/MatMul/ReadVariableOp'^model_5/dense_9/BiasAdd/ReadVariableOp&^model_5/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::2T
(model_5/conv2d_18/BiasAdd/ReadVariableOp(model_5/conv2d_18/BiasAdd/ReadVariableOp2R
'model_5/conv2d_18/Conv2D/ReadVariableOp'model_5/conv2d_18/Conv2D/ReadVariableOp2T
(model_5/conv2d_19/BiasAdd/ReadVariableOp(model_5/conv2d_19/BiasAdd/ReadVariableOp2R
'model_5/conv2d_19/Conv2D/ReadVariableOp'model_5/conv2d_19/Conv2D/ReadVariableOp2T
(model_5/conv2d_20/BiasAdd/ReadVariableOp(model_5/conv2d_20/BiasAdd/ReadVariableOp2R
'model_5/conv2d_20/Conv2D/ReadVariableOp'model_5/conv2d_20/Conv2D/ReadVariableOp2R
'model_5/dense_10/BiasAdd/ReadVariableOp'model_5/dense_10/BiasAdd/ReadVariableOp2P
&model_5/dense_10/MatMul/ReadVariableOp&model_5/dense_10/MatMul/ReadVariableOp2P
&model_5/dense_9/BiasAdd/ReadVariableOp&model_5/dense_9/BiasAdd/ReadVariableOp2N
%model_5/dense_9/MatMul/ReadVariableOp%model_5/dense_9/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_7

a
E__inference_flatten_6_layer_call_and_return_conditional_losses_232336

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ Т 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџxx:& "
 
_user_specified_nameinputs
Х
Ћ
*__inference_conv2d_18_layer_call_fn_231990

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_2319822
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
щ
н
D__inference_dense_10_layer_call_and_return_conditional_losses_232369

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Х
Ћ
*__inference_conv2d_19_layer_call_fn_232011

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_2320032
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
щ
о
E__inference_conv2d_18_layer_call_and_return_conditional_losses_231982

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
3

C__inference_model_5_layer_call_and_return_conditional_losses_232260

inputs,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identityЂ conv2d_18/BiasAdd/ReadVariableOpЂconv2d_18/Conv2D/ReadVariableOpЂ conv2d_19/BiasAdd/ReadVariableOpЂconv2d_19/Conv2D/ReadVariableOpЂ conv2d_20/BiasAdd/ReadVariableOpЂconv2d_20/Conv2D/ReadVariableOpЂdense_10/BiasAdd/ReadVariableOpЂdense_10/MatMul/ReadVariableOpЂdense_9/BiasAdd/ReadVariableOpЂdense_9/MatMul/ReadVariableOpГ
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOpС
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx *
paddingSAME*
strides
2
conv2d_18/Conv2DЊ
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOpА
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx 2
conv2d_18/BiasAdd~
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx 2
conv2d_18/ReluГ
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_19/Conv2D/ReadVariableOpз
conv2d_19/Conv2DConv2Dconv2d_18/Relu:activations:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx*
paddingSAME*
strides
2
conv2d_19/Conv2DЊ
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOpА
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
conv2d_19/BiasAdd~
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
conv2d_19/ReluГ
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOpз
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx*
paddingSAME*
strides
2
conv2d_20/Conv2DЊ
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOpА
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџxx2
conv2d_20/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ Т 2
flatten_6/Const
flatten_6/ReshapeReshapeconv2d_20/Relu:activations:0flatten_6/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2
flatten_6/ReshapeЇ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
2*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMulflatten_6/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dense_9/MatMulЄ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_9/BiasAdd/ReadVariableOpЁ
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dense_9/BiasAddp
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dense_9/ReluЈ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02 
dense_10/MatMul/ReadVariableOpЂ
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_10/MatMulЇ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_10/BiasAdd/ReadVariableOpЅ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
dense_10/BiasAddР
IdentityIdentitydense_10/BiasAdd:output:0!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
!
к
C__inference_model_5_layer_call_and_return_conditional_losses_232125
input_7,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2+
'dense_10_statefulpartitionedcall_args_1+
'dense_10_statefulpartitionedcall_args_2
identityЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ dense_10/StatefulPartitionedCallЂdense_9/StatefulPartitionedCallЛ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinput_7(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_2319822#
!conv2d_18/StatefulPartitionedCallо
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_2320032#
!conv2d_19/StatefulPartitionedCallо
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_2320242#
!conv2d_20/StatefulPartitionedCallъ
flatten_6/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:џџџџџџџџџ*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2320512
flatten_6/PartitionedCallФ
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ2*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2320702!
dense_9/StatefulPartitionedCallЯ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0'dense_10_statefulpartitionedcall_args_1'dense_10_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2320922"
 dense_10/StatefulPartitionedCallЎ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:' #
!
_user_specified_name	input_7
ЄJ
Ш
__inference__traced_save_232511
file_prefix/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1Ѕ
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_74f45ee02c334cfe8b96d4946aed0946/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameі
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*
valueўBћ%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*х
_input_shapesг
а: : : : ::::
2:2:2
:
: : : : : : : : : : ::::
2:2:2
:
: : : ::::
2:2:2
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ф
F
*__inference_flatten_6_layer_call_fn_232341

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:џџџџџџџџџ*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2320512
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџxx:& "
 
_user_specified_nameinputs
!
к
C__inference_model_5_layer_call_and_return_conditional_losses_232105
input_7,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2+
'dense_10_statefulpartitionedcall_args_1+
'dense_10_statefulpartitionedcall_args_2
identityЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ dense_10/StatefulPartitionedCallЂdense_9/StatefulPartitionedCallЛ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinput_7(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_2319822#
!conv2d_18/StatefulPartitionedCallо
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_2320032#
!conv2d_19/StatefulPartitionedCallо
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_2320242#
!conv2d_20/StatefulPartitionedCallъ
flatten_6/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:џџџџџџџџџ*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2320512
flatten_6/PartitionedCallФ
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ2*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2320702!
dense_9/StatefulPartitionedCallЯ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0'dense_10_statefulpartitionedcall_args_1'dense_10_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2320922"
 dense_10/StatefulPartitionedCallЎ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:' #
!
_user_specified_name	input_7
!
й
C__inference_model_5_layer_call_and_return_conditional_losses_232183

inputs,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2*
&dense_9_statefulpartitionedcall_args_1*
&dense_9_statefulpartitionedcall_args_2+
'dense_10_statefulpartitionedcall_args_1+
'dense_10_statefulpartitionedcall_args_2
identityЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ dense_10/StatefulPartitionedCallЂdense_9/StatefulPartitionedCallК
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_2319822#
!conv2d_18/StatefulPartitionedCallо
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_2320032#
!conv2d_19/StatefulPartitionedCallо
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџxx*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_2320242#
!conv2d_20/StatefulPartitionedCallъ
flatten_6/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*)
_output_shapes
:џџџџџџџџџ*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_2320512
flatten_6/PartitionedCallФ
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0&dense_9_statefulpartitionedcall_args_1&dense_9_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ2*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2320702!
dense_9/StatefulPartitionedCallЯ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0'dense_10_statefulpartitionedcall_args_1'dense_10_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_2320922"
 dense_10/StatefulPartitionedCallЎ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:џџџџџџџџџxx::::::::::2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs"ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Г
serving_default
C
input_78
serving_default_input_7:0џџџџџџџџџxx<
dense_100
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:Во
Ж<
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
q_default_save_signature
*r&call_and_return_all_conditional_losses
s__call__"9
_tf_keras_modelє8{"class_name": "Model", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 120, 120, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_18", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_19", "inbound_nodes": [[["conv2d_18", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["conv2d_19", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["flatten_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["dense_10", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 120, 120, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_18", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_19", "inbound_nodes": [[["conv2d_18", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["conv2d_19", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["flatten_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["dense_10", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Б"Ў
_tf_keras_input_layer{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 120, 120, 1], "config": {"batch_input_shape": [null, 120, 120, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
э

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"Ш
_tf_keras_layerЎ{"class_name": "Conv2D", "name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
ю

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*v&call_and_return_all_conditional_losses
w__call__"Щ
_tf_keras_layerЏ{"class_name": "Conv2D", "name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
э

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"Ш
_tf_keras_layerЎ{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
А
 regularization_losses
!trainable_variables
"	variables
#	keras_api
*z&call_and_return_all_conditional_losses
{__call__"Ё
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ѕ

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
*|&call_and_return_all_conditional_losses
}__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 115200}}}}
ѕ

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
*~&call_and_return_all_conditional_losses
__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}

0iter

1beta_1

2beta_2
	3decay
4learning_ratem]m^m_m`mamb$mc%md*me+mfvgvhvivjvkvl$vm%vn*vo+vp"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
5
$6
%7
*8
+9"
trackable_list_wrapper
f
0
1
2
3
4
5
$6
%7
*8
+9"
trackable_list_wrapper
З
5non_trainable_variables

6layers
	regularization_losses
7layer_regularization_losses

trainable_variables
	variables
8metrics
s__call__
q_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
*:( 2conv2d_18/kernel
: 2conv2d_18/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

9non_trainable_variables

:layers
regularization_losses
;layer_regularization_losses
trainable_variables
	variables
<metrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_19/kernel
:2conv2d_19/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

=non_trainable_variables

>layers
regularization_losses
?layer_regularization_losses
trainable_variables
	variables
@metrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_20/kernel
:2conv2d_20/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

Anon_trainable_variables

Blayers
regularization_losses
Clayer_regularization_losses
trainable_variables
	variables
Dmetrics
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

Enon_trainable_variables

Flayers
 regularization_losses
Glayer_regularization_losses
!trainable_variables
"	variables
Hmetrics
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
": 
22dense_9/kernel
:22dense_9/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper

Inon_trainable_variables

Jlayers
&regularization_losses
Klayer_regularization_losses
'trainable_variables
(	variables
Lmetrics
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
!:2
2dense_10/kernel
:
2dense_10/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper

Mnon_trainable_variables

Nlayers
,regularization_losses
Olayer_regularization_losses
-trainable_variables
.	variables
Pmetrics
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

	Rtotal
	Scount
T
_fn_kwargs
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
+&call_and_return_all_conditional_losses
__call__"х
_tf_keras_layerЫ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper

Ynon_trainable_variables

Zlayers
Uregularization_losses
[layer_regularization_losses
Vtrainable_variables
W	variables
\metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/:- 2Adam/conv2d_18/kernel/m
!: 2Adam/conv2d_18/bias/m
/:- 2Adam/conv2d_19/kernel/m
!:2Adam/conv2d_19/bias/m
/:-2Adam/conv2d_20/kernel/m
!:2Adam/conv2d_20/bias/m
':%
22Adam/dense_9/kernel/m
:22Adam/dense_9/bias/m
&:$2
2Adam/dense_10/kernel/m
 :
2Adam/dense_10/bias/m
/:- 2Adam/conv2d_18/kernel/v
!: 2Adam/conv2d_18/bias/v
/:- 2Adam/conv2d_19/kernel/v
!:2Adam/conv2d_19/bias/v
/:-2Adam/conv2d_20/kernel/v
!:2Adam/conv2d_20/bias/v
':%
22Adam/dense_9/kernel/v
:22Adam/dense_9/bias/v
&:$2
2Adam/dense_10/kernel/v
 :
2Adam/dense_10/bias/v
ч2ф
!__inference__wrapped_model_231969О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_7џџџџџџџџџxx
к2з
C__inference_model_5_layer_call_and_return_conditional_losses_232300
C__inference_model_5_layer_call_and_return_conditional_losses_232105
C__inference_model_5_layer_call_and_return_conditional_losses_232260
C__inference_model_5_layer_call_and_return_conditional_losses_232125Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
(__inference_model_5_layer_call_fn_232330
(__inference_model_5_layer_call_fn_232161
(__inference_model_5_layer_call_fn_232196
(__inference_model_5_layer_call_fn_232315Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
E__inference_conv2d_18_layer_call_and_return_conditional_losses_231982з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
*__inference_conv2d_18_layer_call_fn_231990з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Є2Ё
E__inference_conv2d_19_layer_call_and_return_conditional_losses_232003з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
*__inference_conv2d_19_layer_call_fn_232011з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Є2Ё
E__inference_conv2d_20_layer_call_and_return_conditional_losses_232024з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
*__inference_conv2d_20_layer_call_fn_232032з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
я2ь
E__inference_flatten_6_layer_call_and_return_conditional_losses_232336Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_6_layer_call_fn_232341Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_9_layer_call_and_return_conditional_losses_232352Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_9_layer_call_fn_232359Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_10_layer_call_and_return_conditional_losses_232369Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_10_layer_call_fn_232376Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
3B1
$__inference_signature_wrapper_232220input_7
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
  
!__inference__wrapped_model_231969{
$%*+8Ђ5
.Ђ+
)&
input_7џџџџџџџџџxx
Њ "3Њ0
.
dense_10"
dense_10џџџџџџџџџ
к
E__inference_conv2d_18_layer_call_and_return_conditional_losses_231982IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 В
*__inference_conv2d_18_layer_call_fn_231990IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ к
E__inference_conv2d_19_layer_call_and_return_conditional_losses_232003IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 В
*__inference_conv2d_19_layer_call_fn_232011IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџк
E__inference_conv2d_20_layer_call_and_return_conditional_losses_232024IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 В
*__inference_conv2d_20_layer_call_fn_232032IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЄ
D__inference_dense_10_layer_call_and_return_conditional_losses_232369\*+/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ

 |
)__inference_dense_10_layer_call_fn_232376O*+/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ
Ѕ
C__inference_dense_9_layer_call_and_return_conditional_losses_232352^$%1Ђ.
'Ђ$
"
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ2
 }
(__inference_dense_9_layer_call_fn_232359Q$%1Ђ.
'Ђ$
"
inputsџџџџџџџџџ
Њ "џџџџџџџџџ2Ћ
E__inference_flatten_6_layer_call_and_return_conditional_losses_232336b7Ђ4
-Ђ*
(%
inputsџџџџџџџџџxx
Њ "'Ђ$

0џџџџџџџџџ
 
*__inference_flatten_6_layer_call_fn_232341U7Ђ4
-Ђ*
(%
inputsџџџџџџџџџxx
Њ "џџџџџџџџџМ
C__inference_model_5_layer_call_and_return_conditional_losses_232105u
$%*+@Ђ=
6Ђ3
)&
input_7џџџџџџџџџxx
p

 
Њ "%Ђ"

0џџџџџџџџџ

 М
C__inference_model_5_layer_call_and_return_conditional_losses_232125u
$%*+@Ђ=
6Ђ3
)&
input_7џџџџџџџџџxx
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Л
C__inference_model_5_layer_call_and_return_conditional_losses_232260t
$%*+?Ђ<
5Ђ2
(%
inputsџџџџџџџџџxx
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Л
C__inference_model_5_layer_call_and_return_conditional_losses_232300t
$%*+?Ђ<
5Ђ2
(%
inputsџџџџџџџџџxx
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 
(__inference_model_5_layer_call_fn_232161h
$%*+@Ђ=
6Ђ3
)&
input_7џџџџџџџџџxx
p

 
Њ "џџџџџџџџџ

(__inference_model_5_layer_call_fn_232196h
$%*+@Ђ=
6Ђ3
)&
input_7џџџџџџџџџxx
p 

 
Њ "џџџџџџџџџ

(__inference_model_5_layer_call_fn_232315g
$%*+?Ђ<
5Ђ2
(%
inputsџџџџџџџџџxx
p

 
Њ "џџџџџџџџџ

(__inference_model_5_layer_call_fn_232330g
$%*+?Ђ<
5Ђ2
(%
inputsџџџџџџџџџxx
p 

 
Њ "џџџџџџџџџ
Џ
$__inference_signature_wrapper_232220
$%*+CЂ@
Ђ 
9Њ6
4
input_7)&
input_7џџџџџџџџџxx"3Њ0
.
dense_10"
dense_10џџџџџџџџџ
