from abc import abstractmethod
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from helix_viz.audio_processor import AudioProcessor


class VisualizerBase(QtWidgets.QWidget):
    def __init__(
        self,
        processor: Optional[AudioProcessor],
        update_interval_ms: int = 50,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.processor = processor

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(update_interval_ms)
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start()

    @abstractmethod
    def update_visualization(self) -> None:
        pass
