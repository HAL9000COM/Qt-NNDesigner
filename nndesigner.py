from typing import Union
from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QWidget,
    QFileDialog,
    QTableWidgetItem,
    QComboBox,
    QMessageBox,
)
from nndesigner_ui import Ui_nndesigner
import logging
import ast
import tensorflow as tf
import os
import importlib
import tf2onnx


# set log level
logging.basicConfig(level=logging.DEBUG)


class infoLogger(logging.Handler):
    def emit(self, record):
        self.edit.append_line(self.format(record))  # type: ignore


class NNDesigner(QWidget, Ui_nndesigner):
    """
    NNdesigner is a widget for designing neural network.
    It can be used to design a neural network and export the network to a file.

    """

    def __init__(self):
        super(NNDesigner, self).__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.tableWidget.dropEvent = self.table_dropEvent
        # self.tableWidget.dragEnterEvent = self.table_dragEnterEvent
        self.lineEdit_export_path.setAcceptDrops(True)
        self.pushButton_add.clicked.connect(self.add)
        self.pushButton_delete.clicked.connect(self.delete)
        self.pushButton_up.clicked.connect(self.up)
        self.pushButton_down.clicked.connect(self.down)
        self.pushButton_test.clicked.connect(self.test)
        self.pushButton_browse.clicked.connect(self.browse)
        self.pushButton_export.clicked.connect(self.export)
        self.lineEdit_export_path.textChanged.connect(self.export_path_change)
        self.pushButton_clear.clicked.connect(self.clear)
        self.dtype = self.comboBox_dtype.currentText()
        self.comboBox_dtype.currentTextChanged.connect(self.dtype_change)

        h = infoLogger()
        h.edit = self
        h.setLevel(logging.INFO)
        logging.getLogger().addHandler(h)

        self.layer_types = [
            "Dense",
            "Activation",
            "Convolution",
            "Pooling",
            "Recurrent",
            "Preprocessing",
            "Normalization",
            "Regularization",
            "Attention",
            "Reshaping",
            "Merging",
            "Embedding",
            "Masking",
        ]
        self.layers_dict = {
            "Dense": ["Dense"],
            "Input Layer": ["Input"],
            "Output Layer": ["Dense"],
            "Activation": [
                "ReLU",
                "LeakyReLU",
                "PReLU",
                "ELU",
                "SELU",
                "ThresholdedReLU",
                "tanh",
                "Softmax",
                "Softplus",
                "Softsign",
                "Sigmoid",
                "Hard Sigmoid",
                "Linear",
                "Exponential",
            ],
            "Convolution": [
                "Conv1D",
                "Conv2D",
                "Conv3D",
                "SeparableConv1D",
                "SeparableConv2D",
                "DepthwiseConv2D",
                "Conv1DTranspose",
                "Conv2DTranspose",
                "Conv3DTranspose",
            ],
            "Pooling": [
                "MaxPooling1D",
                "MaxPooling2D",
                "MaxPooling3D",
                "AveragePooling1D",
                "AveragePooling2D",
                "AveragePooling3D",
                "GlobalMaxPooling1D",
                "GlobalMaxPooling2D",
                "GlobalMaxPooling3D",
                "GlobalAveragePooling1D",
                "GlobalAveragePooling2D",
                "GlobalAveragePooling3D",
            ],
            "Recurrent": [
                "LSTM",
                "GRU",
                "SimpleRNN",
                "TimeDistributed",
                "Bidirectional",
                "ConvLSTM1D",
                "ConvLSTM2D",
                "ConvLSTM3D",
                "Base RNN",
            ],
            "Preprocessing": [
                "TextVectorization",
                "Normalization",
                "Discretization",
                "CategoryEncoding",
                "Hashing",
                "HashedCrossing",
                "StringLookup",
                "IntegerLookup",
                "Resizing",
                "Rescaling",
                "CenterCrop",
                "RandomCrop",
                "RandomFlip",
                "RandomTranslation",
                "RandomRotation",
                "RandomZoom",
                "RandomContrast",
                "RandomBrightness",
            ],
            "Normalization": [
                "BatchNormalization",
                "LayerNormalization",
                "UnitNormalization",
                "GroupNormalization",
            ],
            "Regularization": [
                "Dropout",
                "SpatialDropout1D",
                "SpatialDropout2D",
                "SpatialDropout3D",
                "GaussianDropout",
                "GaussianNoise",
                "ActivityRegularization",
                "AlphaDropout",
            ],
            "Attention": ["MultiHeadAttention", "Attention", "AdditiveAttention"],
            "Reshaping": [
                "Reshape",
                "Flatten",
                "RepeatVector",
                "Permute",
                "Cropping1D",
                "Cropping2D",
                "Cropping3D",
                "UpSampling1D",
                "UpSampling2D",
                "UpSampling3D",
                "ZeroPadding1D",
                "ZeroPadding2D",
                "ZeroPadding3D",
            ],
            "Merging": [
                "Concatenate",
                "Average",
                "Maximum",
                "Minimum",
                "Add",
                "Subtract",
                "Multiply",
                "Dot",
            ],
            "Embedding": ["Embedding"],
            "Masking": ["Masking"],
        }
        self.args_dict = {
            "Input": "shape=(2,); batch_size=None; name=None; dtype=None; sparse=None; tensor=None; ragged=None; type_spec=None",
            "Dense": 'units=1;activation=None;use_bias=True;kernel_initializer="glorot_uniform";bias_initializer="zeros";kernel_regularizer=None;bias_regularizer=None;activity_regularizer=None;kernel_constraint=None;bias_constraint=None',
            # activation
            "SELU": "None",
            "tanh": "None",
            "Softplus": "None",
            "Softsign": "None",
            "Sigmoid": "None",
            "Hard Sigmoid": "None",
            "Linear": "None",
            "Exponential": "None",
            "ReLU": "max_value=None;negative_slope=0.0;threshold=0.0",
            "LeakyReLU": "alpha=0.3",
            "PReLU": 'alpha_initializer="zeros";alpha_regularizer=None;alpha_constraint=None;shared_axes=None',
            "ELU": "alpha=1.0",
            "ThresholdedReLU": "theta=1.0",
            "Softmax": "axis=-1",
            # Convolution
            "Conv1D": 'filters=1;kernel_size=1;strides=1;padding="valid";data_format="channels_last";dilation_rate=1;groups=1;activation=None;use_bias=True;kernel_initializer="glorot_uniform";bias_initializer="zeros";kernel_regularizer=None;bias_regularizer=None;activity_regularizer=None;kernel_constraint=None;bias_constraint=None',
            "Conv2D": 'filters=1;kernel_size=(1, 1);strides=(1, 1);padding="valid";data_format=None;dilation_rate=(1, 1);groups=1;activation=None;use_bias=True;kernel_initializer="glorot_uniform";bias_initializer="zeros";kernel_regularizer=None;bias_regularizer=None;activity_regularizer=None;kernel_constraint=None;bias_constraint=None',
            "Conv3D": 'filters=1;kernel_size=(1, 1, 1);strides=(1, 1, 1);padding="valid";data_format=None;dilation_rate=(1, 1, 1);groups=1;activation=None;use_bias=True;kernel_initializer="glorot_uniform";bias_initializer="zeros";kernel_regularizer=None;bias_regularizer=None;activity_regularizer=None;kernel_constraint=None;bias_constraint=None',
            "SeparableConv1D": 'filters=1;kernel_size=1;strides=1;padding="valid";data_format="channels_last";dilation_rate=1;depth_multiplier=1;activation=None;use_bias=True;depthwise_initializer="glorot_uniform";pointwise_initializer="glorot_uniform";bias_initializer="zeros";depthwise_regularizer=None;pointwise_regularizer=None;bias_regularizer=None;activity_regularizer=None;depthwise_constraint=None;pointwise_constraint=None;bias_constraint=None',
            "SeparableConv2D": 'filters=1;kernel_size=(1, 1);strides=(1, 1);padding="valid";data_format=None;dilation_rate=(1, 1);depth_multiplier=1;activation=None;use_bias=True;depthwise_initializer="glorot_uniform";pointwise_initializer="glorot_uniform";bias_initializer="zeros";depthwise_regularizer=None;pointwise_regularizer=None;bias_regularizer=None;activity_regularizer=None;depthwise_constraint=None;pointwise_constraint=None;bias_constraint=None',
            "DepthwiseConv2D": 'kernel_size=(1, 1);strides=(1, 1);padding="valid";data_format=None;dilation_rate=(1, 1);depth_multiplier=1;activation=None;use_bias=True;depthwise_initializer="glorot_uniform";bias_initializer="zeros";depthwise_regularizer=None;bias_regularizer=None;activity_regularizer=None;depthwise_constraint=None;bias_constraint=None',
            "Conv1DTranspose": 'filters=1;kernel_size=1;strides=1;padding="valid";output_padding=None;data_format="channels_last";dilation_rate=1;activation=None;use_bias=True;kernel_initializer="glorot_uniform";bias_initializer="zeros";kernel_regularizer=None;bias_regularizer=None;activity_regularizer=None;kernel_constraint=None;bias_constraint=None',
            "Conv2DTranspose": 'filters=1;kernel_size=(1, 1);strides=(1, 1);padding="valid";output_padding=None;data_format=None;dilation_rate=(1, 1);activation=None;use_bias=True;kernel_initializer="glorot_uniform";bias_initializer="zeros";kernel_regularizer=None;bias_regularizer=None;activity_regularizer=None;kernel_constraint=None;bias_constraint=None',
            "Conv3DTranspose": 'filters=1;kernel_size=(1, 1, 1);strides=(1, 1, 1);padding="valid";output_padding=None;data_format=None;dilation_rate=(1, 1, 1);activation=None;use_bias=True;kernel_initializer="glorot_uniform";bias_initializer="zeros";kernel_regularizer=None;bias_regularizer=None;activity_regularizer=None;kernel_constraint=None;bias_constraint=None',
            # Pooling
            "MaxPooling1D": 'pool_size=2;strides=None;padding="valid";data_format="channels_last"',
            "MaxPooling2D": 'pool_size=(2, 2);strides=None;padding="valid";data_format=None',
            "MaxPooling3D": 'pool_size=(2, 2, 2);strides=None;padding="valid";data_format=None',
            "AveragePooling1D": 'pool_size=2;strides=None;padding="valid";data_format="channels_last"',
            "AveragePooling2D": 'pool_size=(2, 2);strides=None;padding="valid";data_format=None',
            "AveragePooling3D": 'pool_size=(2, 2, 2);strides=None;padding="valid";data_format=None',
            "GlobalMaxPooling1D": 'data_format="channels_last"; keepdims=False',
            "GlobalMaxPooling2D": "data_format=None; keepdims=False",
            "GlobalMaxPooling3D": "data_format=None; keepdims=False",
            "GlobalAveragePooling1D": 'data_format="channels_last"; keepdims=False',
            "GlobalAveragePooling2D": "data_format=None; keepdims=False",
            "GlobalAveragePooling3D": "data_format=None; keepdims=False",
            # Recurrent
            "LSTM": 'units=1; activation="tanh"; recurrent_activation="sigmoid"; use_bias=True; kernel_initializer="glorot_uniform"; recurrent_initializer="orthogonal"; bias_initializer="zeros"; unit_forget_bias=True; kernel_regularizer=None; recurrent_regularizer=None; bias_regularizer=None; activity_regularizer=None; kernel_constraint=None; recurrent_constraint=None; bias_constraint=None; dropout=0.0; recurrent_dropout=0.0; return_sequences=False; return_state=False; go_backwards=False; stateful=False; time_major=False; unroll=False',
            "GRU": 'units=1; activation="tanh"; recurrent_activation="sigmoid"; use_bias=True; kernel_initializer="glorot_uniform"; recurrent_initializer="orthogonal"; bias_initializer="zeros"; kernel_regularizer=None; recurrent_regularizer=None; bias_regularizer=None; activity_regularizer=None; kernel_constraint=None; recurrent_constraint=None; bias_constraint=None; dropout=0.0; recurrent_dropout=0.0; return_sequences=False; return_state=False; go_backwards=False; stateful=False; unroll=False; time_major=False; reset_after=True',
            "SimpleRNN": 'units; activation="tanh"; use_bias=True; kernel_initializer="glorot_uniform"; recurrent_initializer="orthogonal"; bias_initializer="zeros"; kernel_regularizer=None; recurrent_regularizer=None; bias_regularizer=None; activity_regularizer=None; kernel_constraint=None; recurrent_constraint=None; bias_constraint=None; dropout=0.0; recurrent_dropout=0.0; return_sequences=False; return_state=False; go_backwards=False; stateful=False; unroll=False',
            "TimeDistributed": "None",
            "Bidirectional": "None",
            "ConvLSTM1D": 'filters=1; kernel_size=1; strides=1; padding="valid"; data_format=None; dilation_rate=1; activation="tanh"; recurrent_activation="hard_sigmoid"; use_bias=True; kernel_initializer="glorot_uniform"; recurrent_initializer="orthogonal"; bias_initializer="zeros"; unit_forget_bias=True; kernel_regularizer=None; recurrent_regularizer=None; bias_regularizer=None; activity_regularizer=None; kernel_constraint=None; recurrent_constraint=None; bias_constraint=None; return_sequences=False; return_state=False; go_backwards=False; stateful=False; dropout=0.0; recurrent_dropout=0.0',
            "ConvLSTM2D": 'filters=1; kernel_size=(1, 1); strides=(1, 1); padding="valid"; data_format=None; dilation_rate=(1, 1); activation="tanh"; recurrent_activation="hard_sigmoid"; use_bias=True; kernel_initializer="glorot_uniform"; recurrent_initializer="orthogonal"; bias_initializer="zeros"; unit_forget_bias=True; kernel_regularizer=None; recurrent_regularizer=None; bias_regularizer=None; activity_regularizer=None; kernel_constraint=None; recurrent_constraint=None; bias_constraint=None; return_sequences=False; return_state=False; go_backwards=False; stateful=False; dropout=0.0; recurrent_dropout=0.0',
            "ConvLSTM3D": 'filters=1; kernel_size=(1, 1, 1); strides=(1, 1, 1); padding="valid"; data_format=None; dilation_rate=(1, 1, 1); activation="tanh"; recurrent_activation="hard_sigmoid"; use_bias=True; kernel_initializer="glorot_uniform"; recurrent_initializer="orthogonal"; bias_initializer="zeros"; unit_forget_bias=True; kernel_regularizer=None; recurrent_regularizer=None; bias_regularizer=None; activity_regularizer=None; kernel_constraint=None; recurrent_constraint=None; bias_constraint=None; return_sequences=False; return_state=False; go_backwards=False; stateful=False; dropout=0.0; recurrent_dropout=0.0',
            "Base RNN": "None",
            # Preprocessing
            "TextVectorization": 'max_tokens=None; standardize="lower_and_strip_punctuation"; split="whitespace"; ngrams=None; output_mode="int"; output_sequence_length=None; pad_to_max_tokens=False; vocabulary=None; idf_weights=None; sparse=False; ragged=False; encoding="utf-8"',
            "Normalization": "axis=-1; mean=None; variance=None; invert=False",
            "Discretization": 'bin_boundaries=None; num_bins=None; epsilon=0.01; output_mode="int"; sparse=False',
            "CategoryEncoding": 'num_tokens=None; output_mode="multi_hot"; sparse=False',
            "Hashing": 'num_bins=3; mask_value=None; salt=None; output_mode="int"; sparse=False',
            "HashedCrossing": 'num_bins=3; output_mode="int"; sparse=False',
            "StringLookup": 'max_tokens=None; num_oov_indices=1; mask_token=None; oov_token="[UNK]"; vocabulary=None; idf_weights=None; encoding="utf-8"; invert=False; output_mode="int"; sparse=False; pad_to_max_tokens=False',
            "IntegerLookup": 'max_tokens=None; num_oov_indices=1; mask_token=None; oov_token=-1; vocabulary=None; vocabulary_dtype="int64"; idf_weights=None; invert=False; output_mode="int"; sparse=False; pad_to_max_tokens=False',
            "Resizing": 'height=224; width=224; interpolation="bilinear"; crop_to_aspect_ratio=False',
            "Rescaling": "scale=1.0; offset=0.0",
            "CenterCrop": "height=224; width=224",
            "RandomCrop": "height=224; width=224; seed=None",
            "RandomFlip": 'mode="horizontal_and_vertical"; seed=None',
            "RandomTranslation": 'height_factor=(-0.1, 0.1); width_factor=(-0.1, 0.1); fill_mode="reflect"; interpolation="bilinear"; seed=None; fill_value=0.0',
            "RandomRotation": 'factor=(-0.1, 0.1); fill_mode="reflect"; interpolation="bilinear"; seed=None; fill_value=0.0',
            "RandomZoom": 'height_factor=(-0.1, 0.1); width_factor=None; fill_mode="reflect"; interpolation="bilinear"; seed=None; fill_value=0.0',
            "RandomContrast": "factor=(-0.1, 0.1); seed=None",
            "RandomBrightness": "factor=(-0.1, 0.1); value_range=(0, 255); seed=None",
            # Normalization
            "BatchNormalization": 'axis=-1; momentum=0.99; epsilon=0.001; center=True; scale=True; beta_initializer="zeros"; gamma_initializer="ones"; moving_mean_initializer="zeros"; moving_variance_initializer="ones"; beta_regularizer=None; gamma_regularizer=None; beta_constraint=None; gamma_constraint=None; synchronized=False',
            "LayerNormalization": 'axis=-1; epsilon=0.001; center=True; scale=True; beta_initializer="zeros"; gamma_initializer="ones"; beta_regularizer=None; gamma_regularizer=None; beta_constraint=None; gamma_constraint=None',
            "UnitNormalization": "axis=-1",
            "GroupNormalization": 'groups=32; axis=-1; epsilon=0.001; center=True; scale=True; beta_initializer="zeros"; gamma_initializer="ones"; beta_regularizer=None; gamma_regularizer=None; beta_constraint=None; gamma_constraint=None',
            # Regularization
            "Dropout": "rate=0.5; noise_shape=None; seed=None",
            "SpatialDropout1D": "rate=0.5",
            "SpatialDropout2D": "rate=0.5; data_format=None",
            "SpatialDropout3D": "rate=0.5; data_format=None",
            "GaussianDropout": "rate=0.5; seed=None",
            "GaussianNoise": "stddev=1.0; seed=None",
            "ActivityRegularization": "l1=0.0; l2=0.0",
            "AlphaDropout": "rate=0.5; noise_shape=None; seed=None",
            # Attention
            "MultiHeadAttention": 'num_heads=1; key_dim=None; value_dim=None; dropout=0.0; use_bias=True; output_shape=None; attention_axes=None; kernel_initializer="glorot_uniform"; bias_initializer="zeros"; kernel_regularizer=None; bias_regularizer=None; activity_regularizer=None; kernel_constraint=None; bias_constraint=None',
            "Attention": 'use_scale=False; score_mode="dot"',
            "AdditiveAttention": "use_scale=False",
            # Reshape
            "Reshape": "target_shape=(1,1)",
            "Flatten": "data_format=None",
            "RepeatVector": "n=1",
            "Permute": "dims=(2, 1)",
            "Cropping1D": "cropping=(1, 1); data_format=None",
            "Cropping2D": "cropping=((0, 0), (0, 0)); data_format=None",
            "Cropping3D": "cropping=((1, 1), (1, 1), (1, 1)); data_format=None",
            "UpSampling1D": "size=2",
            "UpSampling2D": 'size=(2, 2); data_format=None; interpolation="nearest"',
            "UpSampling3D": 'size=(2, 2, 2); data_format=None; interpolation="nearest"',
            "ZeroPadding1D": "padding=1",
            "ZeroPadding2D": "padding=(1, 1); data_format=None",
            "ZeroPadding3D": "padding=(1, 1, 1); data_format=None",
            # Merge
            "Concatenate": "axis=-1",
            "Average": "None",
            "Maximum": "None",
            "Minimum": "None",
            "Add": "None",
            "Subtract": "None",
            "Multiply": "None",
            "Dot": "axes=-1",
            # Embedding
            "Embedding": 'input_dim=1; output_dim=1; embeddings_initializer="uniform"; embeddings_regularizer=None; activity_regularizer=None; embeddings_constraint=None; mask_zero=False; input_length=None; sparse=False',
            # Masking
            "Masking": "mask_value=0.0",
        }

        # set the first row
        self.add_row_io(0)
        self.add_row_io(1)

    def add_row_io(self, row):
        combo = QComboBox()
        text = self.tableWidget.item(row, 0).text()
        combo.addItems(self.layers_dict[text])
        self.tableWidget.setCellWidget(row, 1, combo)
        item = QTableWidgetItem(self.args_dict[combo.currentText()])
        self.tableWidget.setItem(row, 2, item)

    def add(self):
        # get current selected row
        row = self.tableWidget.currentRow()
        if row == -1:
            row = 0
        # insert a new row
        if row == self.tableWidget.rowCount() - 1:
            return
        self.tableWidget.insertRow(row + 1)
        # set the item
        combo = QComboBox()
        combo.addItems(self.layer_types)
        self.tableWidget.setCellWidget(row + 1, 0, combo)

        self.tableWidget.cellWidget(row + 1, 0).currentTextChanged.connect(self.layer_change)  # type: ignore
        text = self.tableWidget.cellWidget(row + 1, 0).currentText()  # type: ignore
        # print(combo.pos())

        # self.tableWidget.cellWidget(row + 1, 0).setCurrentIndex(1)  # type: ignore
        # print(combo.pos())
        # self.tableWidget.cellWidget(row + 1, 0).setCurrentIndex(0)  # type: ignore #trigger twice to setback
        # print(combo.pos())

        ## init the row, should be done using the signal connection
        row += 1
        combo = QComboBox()
        combo.addItems(self.layers_dict[text])
        self.tableWidget.setCellWidget(row, 1, combo)
        self.tableWidget.cellWidget(row, 1).currentTextChanged.connect(self.set_args)  # type: ignore
        self.set_args(self.tableWidget.cellWidget(row, 1).currentText())  # type: ignore
        item = QTableWidgetItem(self.args_dict[self.tableWidget.cellWidget(row, 1).currentText()])  # type: ignore
        self.tableWidget.setItem(row, 2, item)

        logging.debug("add a new row at Row " + str(row + 1))

    def delete(self):
        # get current selected row
        row = self.tableWidget.currentRow()
        if row == -1:  # no selected row
            return
        # delete the row
        if (
            row == 0 or row == self.tableWidget.rowCount() - 1
        ):  # can't delete the first and last row
            return
        self.tableWidget.removeRow(row)
        logging.debug("delete row at Row " + str(row))

    def dtype_change(self):
        self.dtype = self.comboBox_dtype.currentText()

    def layer_change(self, text):
        combo_box = self.sender()
        row = self.tableWidget.indexAt(combo_box.pos()).row()  # type: ignore
        try:
            combo = QComboBox()
            combo.addItems(self.layers_dict[text])
            self.tableWidget.setCellWidget(row, 1, combo)
            self.tableWidget.cellWidget(row, 1).currentTextChanged.connect(self.set_args)  # type: ignore
            self.set_args(self.tableWidget.cellWidget(row, 1).currentText())  # type: ignore
        except KeyError:
            logging.debug("no layers for " + text)
        logging.debug("change layer type to " + text)

    def set_args(self, text):
        combo_box = self.sender()
        row = self.tableWidget.indexAt(combo_box.pos()).row()  # type: ignore
        try:
            item = QTableWidgetItem(self.args_dict[text])
        except KeyError:
            item = QTableWidgetItem("None")
            logging.debug("no args for " + text)
        if item.text() == "None":
            item.setFlags(~Qt.ItemIsEditable)  # type: ignore
        self.tableWidget.setItem(row, 2, item)

    def up(self):
        row = self.tableWidget.currentRow()

        if row == -1:  # no selected row
            return
        if (
            row == 0 or row == 1 or row == self.tableWidget.rowCount() - 1
        ):  # can't move the first, 2nd and last row
            return

        # exchange the row
        self.tableWidget.insertRow(row - 1)

        for i in range(self.tableWidget.columnCount()):
            if self.tableWidget.cellWidget(row + 1, i):  # move the widget
                widget_set = self.tableWidget.cellWidget(row + 1, i)
                self.tableWidget.setCellWidget(row - 1, i, widget_set)
            else:  # move the item
                self.tableWidget.setItem(
                    row - 1, i, self.tableWidget.takeItem(row + 1, i)
                )

        self.tableWidget.removeRow(row + 1)
        self.tableWidget.selectRow(row - 1)

    def down(self):
        row = self.tableWidget.currentRow()

        if row == -1:
            return  # no selected row
        if (
            row == 0
            or row == self.tableWidget.rowCount() - 2
            or row == self.tableWidget.rowCount() - 1
        ):
            return  # can't move the first, 2nd and last row

        # exchange the row
        self.tableWidget.insertRow(row + 2)

        for i in range(self.tableWidget.columnCount()):
            if self.tableWidget.cellWidget(row, i):
                widget_set = self.tableWidget.cellWidget(row, i)
                self.tableWidget.setCellWidget(row + 2, i, widget_set)
            else:
                self.tableWidget.setItem(row + 2, i, self.tableWidget.takeItem(row, i))

        self.tableWidget.removeRow(row)
        self.tableWidget.selectRow(row + 1)

    def test(self):
        self.model = tf.keras.Sequential()
        self.acti_count = 0
        # self.dense_count = 0
        # self.conv_count = 0
        # self.pool_count = 0
        # self.recur_count = 0
        # self.norm_count = 0
        # self.regu_count = 0
        # self.att_count = 0
        # self.reshape_count = 0
        # self.merge_count = 0
        # self.emb_count = 0
        # self.mask_count = 0
        # read the table
        for i in range(self.tableWidget.rowCount()):
            try:
                text = self.tableWidget.cellWidget(i, 0).currentText()  # type: ignore
            except AttributeError:
                text = self.tableWidget.item(i, 0).text()
            layer_type_name = self.tableWidget.cellWidget(i, 1).currentText()  # type: ignore
            layer_args_str = self.tableWidget.item(i, 2).text()  # type: ignore
            if layer_args_str != "None":
                kwargs = self.str2dict(layer_args_str)
            else:
                kwargs = {}
            kwargs["dtype"] = kwargs.get("dtype", self.dtype)
            tf.keras.mixed_precision.set_global_policy(kwargs["dtype"])
            logging.debug(kwargs)
            logging.debug("layer type: " + text)
            logging.debug("layer name: " + layer_type_name)

            if text == "Activation":
                acti = self.keras_get_activation(
                    layer_type_name, self.acti_count, **kwargs
                )
                self.model.add(acti)
            else:
                try:
                    layer = self.keras_get_layer(layer_type_name, **kwargs)
                    self.model.add(layer)
                except Exception as e:
                    logging.error(e)
                    return

        self.model.build()
        self.model.summary(print_fn=logging.info)  # type: ignore

    def str2dict(self, instr: str) -> dict:
        kwargs = {}
        # delete trailing ;
        # if instr[-1] == ";":
        #     instr = instr[:-1]
        for pair in instr.split(";"):
            key, value = pair.split("=")
            value = value.strip().replace('"', "").strip()

            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass

            if value == "-1":
                value = -1
            elif isinstance(value, str):
                value = value.lower()
                if value == "none":
                    value = None
                elif value == "false":
                    value = False
                elif value == "true":
                    value = True
                elif "." in value:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                elif value.isdigit():
                    value = int(value)

            kwargs[key.strip()] = value

        return kwargs

    def export(self):
        self.test()
        model = self.model
        if self.lineEdit_export_path.text() == "":
            logging.warning("Please enter a valid path")
            QMessageBox.warning(
                self, "Warning", "Please enter a valid path", QMessageBox.Ok
            )
            return
        if self.lineEdit_export_name.text() == "":
            logging.warning("Please enter a valid name")
            QMessageBox.warning(
                self, "Warning", "Please enter a valid name", QMessageBox.Ok
            )
            return
        match self.comboBox_type.currentText():
            case "Keras HDF5":
                model_type = "h5"
            case "ONNX":
                model_type = "onnx"
            case "Keras SavedModel":
                model_type = "savedmodel"
            case _:
                logging.debug(
                    f"model type: {self.comboBox_type.currentText()}not supported"
                )
                logging.warning("Enter a valid model type")
                return

        output_path = self.save_model(
            model,
            self.lineEdit_export_path.text(),
            model_name=self.lineEdit_export_name.text(),
            model_type=model_type,
        )
        logging.info("Model exported to " + output_path)

    def browse(self):
        fpath = QFileDialog.getExistingDirectory(None, "Select export folder", "/")
        self.lineEdit_export_path.setText(fpath)

    def export_path_change(self):
        logging.info("Selected export folder:" + self.lineEdit_export_path.text())

    def clear(self):
        rowcount = self.tableWidget.rowCount()
        for i in range(rowcount - 2):
            self.tableWidget.removeRow(1)
        self.add_row_io(0)
        self.add_row_io(1)
        self.textBrowser.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            # decide where is the mouse position
            if self.lineEdit_export_path.underMouse():
                self.lineEdit_export_path.setText(url.toLocalFile())

    def table_dropEvent(self, event):
        logging.debug(event.position())
        event.ignore()
        pass

    def append_line(self, text: str):
        self.textBrowser.append(text)

    def save_model(self, model, path, **kwargs):
        model_type = kwargs.get("model_type", "onnx")
        model_name = kwargs.get("model_name", model.name)
        match model_type:
            case "onnx":
                output_path = os.path.join(path, model_name + ".onnx")
                model_proto, _ = tf2onnx.convert.from_keras(
                    model, output_path=output_path
                )
            case "savedmodel":
                # output_path=os.path.join(path, model.name+".h5")
                output_path = os.path.join(path, model_name)
                model.save(output_path)
            case "h5":
                output_path = os.path.join(path, model_name + ".h5")
                # change separator
                model.save(output_path)
            case _:
                logging.debug(f"model_type{model_type}not found")
                raise ValueError("model_type must be onnx or h5 or savedmodel")
        output_path = output_path.replace("\\", "/")
        return output_path

    def keras_get_activation(self, layer_type_name="ReLU", layer_count=0, **kwargs):
        # activation_list =[
        #         "ReLU",
        #         "LeakyReLU",
        #         "PReLU",
        #         "ELU",
        #         "SELU",
        #         "ThresholdedReLU",
        #         "tanh",
        #         "Softmax",
        #         "Softplus",
        #         "Softsign",
        #         "Sigmoid",
        #         "Hard Sigmoid",
        #         "Linear",
        #         "Exponential",
        # ]
        from tensorflow.keras import activations  # type: ignore
        from tensorflow.keras.layers import Activation  # type: ignore

        list_noargs = [
            "SELU",
            "tanh",
            "Softplus",
            "Softsign",
            "Sigmoid",
            "Hard Sigmoid",
            "Linear",
            "Exponential",
        ]
        list_args = [
            "ReLU",
            "LeakyReLU",
            "PReLU",
            "ELU",
            "ThresholdedReLU",
            "Softmax",
        ]

        if layer_type_name in list_args:
            try:
                acti = activations.get(layer_type_name)
                acti.__init__(**kwargs)
            except:
                logging.debug("Input error", exc_info=True)
                return
        elif layer_type_name in list_noargs:
            match layer_type_name:
                case "SELU":
                    acti = Activation(
                        activations.selu, name="SELU" + "_" + str(layer_count)
                    )
                case "tanh":
                    acti = Activation(
                        activations.tanh, name="tanh" + "_" + str(layer_count)
                    )
                case "Softplus":
                    acti = Activation(
                        activations.softplus, name="Softplus" + "_" + str(layer_count)
                    )

                case "Softsign":
                    acti = Activation(
                        activations.softsign, name="Softsign" + "_" + str(layer_count)
                    )

                case "Sigmoid":
                    acti = Activation(
                        activations.sigmoid, name="Sigmoid" + "_" + str(layer_count)
                    )

                case "Hard Sigmoid":
                    acti = Activation(
                        activations.hard_sigmoid,
                        name="Hard_Sigmoid" + "_" + str(layer_count),
                    )

                case "Linear":
                    acti = Activation(
                        activations.linear, name="Linear" + "_" + str(layer_count)
                    )

                case "Exponential":
                    acti = Activation(
                        activations.exponential,
                        name="Exponential" + "_" + str(layer_count),
                    )

                case _:
                    logging.debug("Input error", exc_info=True)
                    return
        else:
            logging.debug("Input error", exc_info=True)
            return
        return acti

    def keras_get_layer(self, layer_type_name, **kwargs):
        module_name = f"tensorflow.keras.layers"
        class_name = layer_type_name
        try:
            module = importlib.import_module(module_name)
            layer_class = getattr(module, class_name)
        except (ImportError, AttributeError):
            logging.debug(f"Layer type {layer_type_name} not supported")
            return None

        return layer_class(**kwargs)
