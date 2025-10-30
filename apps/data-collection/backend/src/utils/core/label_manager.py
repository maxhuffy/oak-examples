class LabelManager:
    """
    Manages synchronization of label data between the detection filter
    and the annotation node.

    Handles assigning numeric label IDs used by the neural network and
    mapping them to human-readable class names for visualization.
    """

    def __init__(self, det_filter, annotation_node):
        self.det_filter = det_filter
        self.annotation_node = annotation_node

    def update_labels(self, label_names, offset=0):
        self.det_filter.setLabels(
            labels=[i for i in range(offset, offset + len(label_names))], keep=True
        )
        self.annotation_node.setLabelEncoding(
            {offset + k: v for k, v in enumerate(label_names)}
        )
