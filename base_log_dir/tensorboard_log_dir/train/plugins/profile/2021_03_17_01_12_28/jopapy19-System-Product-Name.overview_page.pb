�	��BtT��@��BtT��@!��BtT��@	(yإ�B?(yإ�B?!(yإ�B?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��BtT��@E����?A'���A��@Y�a�[>��?*	�E���3A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�|@�É�@!������X@)�|@�É�@1������X@:Preprocessing2F
Iterator::Model#ظ�]��?! �Lld`?)�B�=��?1��!{_cU?:Preprocessing2P
Iterator::Model::PrefetchI*S�AЁ?!0����F?)I*S�AЁ?10����F?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�+H3Ɖ�@!g�'7��X@)�Ry=�d?1n>(F�Y*?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	E����?E����?!E����?      ��!       "      ��!       *      ��!       2	'���A��@'���A��@!'���A��@:      ��!       B      ��!       J	�a�[>��?�a�[>��?!�a�[>��?R      ��!       Z	�a�[>��?�a�[>��?!�a�[>��?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 