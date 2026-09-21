"""PySide6 chat window. Adapts LoopEvents to Qt signals; the loop runs on a worker thread."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from laya_agent.agent.loop import AgentLoop, build_loop
from laya_agent.config import Config
from laya_agent.debug import annotate_png, element_listing
from laya_agent.models import InputRequest, StepResult

BOX_COLORS = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#5ac8fa"]


class QtEvents(QObject):
    """LoopEvents implementation. Called from the worker thread; emits to the GUI thread."""

    log = Signal(str)
    step = Signal(object)
    input_needed = Signal(object)
    finished = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._reply_ready = threading.Event()
        self._reply = ""

    def on_log(self, msg: str) -> None:
        self.log.emit(msg)

    def on_step(self, result: StepResult) -> None:
        self.step.emit(result)

    def on_input_needed(self, req: InputRequest) -> str:
        self._reply_ready.clear()
        self.input_needed.emit(req)
        self._reply_ready.wait()
        return self._reply

    def on_finished(self, summary: str) -> None:
        self.finished.emit(summary)

    def reply(self, text: str) -> None:
        self._reply = text
        self._reply_ready.set()


class Preloader(QThread):
    """Loads the model right after the window opens so the first goal is not the slow one."""

    log = Signal(str)

    def __init__(self, loop: AgentLoop) -> None:
        super().__init__()
        self.loop = loop

    def run(self) -> None:
        try:
            self.loop.policy.preload(self.log.emit)
        except Exception as e:  # pragma: no cover
            self.log.emit(f"model load failed: {e}")


class Worker(QThread):
    answered = Signal(str, float)

    def __init__(self, loop: AgentLoop, text: str) -> None:
        super().__init__()
        self.loop = loop
        self.text = text

    def run(self) -> None:
        if self.text.startswith("?"):
            q = self.text[1:].strip()
            self.answered.emit(q, self.loop.answer(q))
        else:
            self.loop.run(self.text)


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.events = QtEvents()
        self.loop = build_loop(cfg, self.events)
        self.worker: Worker | None = None
        self.awaiting_input = False
        self.last_png: bytes | None = None

        self.setWindowTitle("Laya screen agent")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(1100, 640)
        self._build()
        self._wire()
        self.loop.own_hwnd = int(self.winId())
        self._say("system", "Type a goal, e.g. <i>open the Edit menu</i>. Start with <b>?</b> to ask a yes/no question about the screen. Focus the target app before pressing Enter.")
        self._say("system", "loading model in the background...")
        self.preloader = Preloader(self.loop)
        self.preloader.log.connect(lambda m: self._say("system", m))
        self.preloader.start()

    # -- layout ---------------------------------------------------------------
    def _build(self) -> None:
        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.input = QLineEdit()
        self.input.setPlaceholderText("goal, hint, option number, or ? question")
        self.send_btn = QPushButton("Send")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.all_btn = QPushButton("All elements")
        self.all_btn.setToolTip("Parse the target window now and show every element, unpruned")
        row = QHBoxLayout()
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.all_btn)
        lv.addWidget(self.chat, 1)
        lv.addLayout(row)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.preview = QLabel("screenshot appears here")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(480, 300)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["option", "p"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 420)
        self.table.setMaximumHeight(190)
        rv.addWidget(self.preview, 1)
        rv.addWidget(self.table)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 680])
        self.setCentralWidget(split)

    def _wire(self) -> None:
        self.send_btn.clicked.connect(self._submit)
        self.input.returnPressed.connect(self._submit)
        self.stop_btn.clicked.connect(self._stop)
        self.all_btn.clicked.connect(self._show_all)
        self.events.log.connect(self._on_log)
        self.events.step.connect(self._on_step)
        self.events.input_needed.connect(self._on_input_needed)
        self.events.finished.connect(self._on_finished)

    # -- chat helpers -----------------------------------------------------------
    def _say(self, who: str, html: str) -> None:
        color = {"you": "#0a84ff", "agent": "#1c1c1e", "system": "#8e8e93"}.get(who, "#000")
        self.chat.append(f'<div style="margin:4px 0"><b style="color:{color}">{who}</b>: {html}</div>')

    # -- slots ------------------------------------------------------------------
    @Slot()
    def _submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._say("you", text)
        if self.awaiting_input:
            self.awaiting_input = False
            self.events.reply(text)
            return
        if self.worker is not None and self.worker.isRunning():
            self._say("system", "busy. Press Stop first.")
            return
        self.stop_btn.setEnabled(True)
        self.worker = Worker(self.loop, text)
        self.worker.answered.connect(self._on_answered)
        self.worker.finished.connect(lambda: self.stop_btn.setEnabled(False))
        self.worker.start()

    @Slot()
    def _stop(self) -> None:
        self.loop.stop()
        if self.awaiting_input:
            self.awaiting_input = False
            self.events.reply("stop")

    @Slot()
    def _show_all(self) -> None:
        """Parse now (or reuse the last step) and show the unpruned overlay plus the full list."""
        if self.worker is not None and self.worker.isRunning():
            snap, dec = self.loop.last_snapshot, self.loop.last_decision
            if snap is None:
                self._say("system", "no snapshot yet")
                return
        else:
            snap, dec = self.loop.inspect(), None
        shortlist = set(dec.raw.get("_shortlist", [])) if dec else set()
        fine = set(dec.raw.get("_fine", [])) if dec else set()
        chosen = dec.element.id if dec and dec.element else None
        self._show_png(annotate_png(snap, shortlist, fine, chosen))
        listing = element_listing(snap, shortlist, fine).replace("&", "&amp;").replace("<", "&lt;")
        self._say("system", f"<pre style='font-size:11px'>{listing}</pre>")
        self.table.setRowCount(0)

    @Slot(str)
    def _on_log(self, msg: str) -> None:
        self._say("agent", msg)

    @Slot(object)
    def _on_step(self, result: StepResult) -> None:
        self._render(result)
        if result.executed:
            self._say("agent", f"→ {result.note}")

    @Slot(object)
    def _on_input_needed(self, req: InputRequest) -> None:
        self.awaiting_input = True
        if req.kind == "text":
            self._say("agent", "<b>What should I type?</b> Send the exact text.")
            return
        lines = "".join(f"<br>&nbsp;&nbsp;<b>{i}</b>. {label} ({p:.2f})" for i, (label, p) in enumerate(req.options, 1))
        self._say("agent", f"<b>{req.reason}.</b> Pick a number, give a hint, or say stop.{lines}")
        self.activateWindow()

    @Slot(str)
    def _on_finished(self, summary: str) -> None:
        self._say("agent", f"<b>{summary}</b>")

    @Slot(str, float)
    def _on_answered(self, question: str, p: float) -> None:
        self._say("agent", f"P(yes) = <b>{p:.2f}</b> for “{question}”")

    # -- rendering --------------------------------------------------------------
    def _show_png(self, png: bytes) -> None:
        pix = QPixmap()
        pix.loadFromData(png, "PNG")
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def _render(self, result: StepResult) -> None:
        """Grey = every parsed element, orange = shortlisted, green = fine pass, red = chosen.
        Then the top-5 ranks are drawn big on top."""
        d = result.decision
        base = annotate_png(
            result.snapshot,
            set(d.raw.get("_shortlist", [])),
            set(d.raw.get("_fine", [])),
            d.element.id if d.element else None,
        )
        pix = QPixmap()
        pix.loadFromData(base, "PNG")
        elements = d.raw.get("_elements", {})
        painter = QPainter(pix)
        painter.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        for rank, action in enumerate(d.top_actions[:5]):
            el = elements.get(action)
            if el is None:
                continue
            color = QColor(BOX_COLORS[rank])
            painter.setPen(QPen(color, 4))
            painter.drawRect(el.rect.left, el.rect.top, el.rect.width, el.rect.height)
            painter.drawText(el.rect.right + 4, max(24, el.rect.top + 20), f"#{rank + 1}")
        painter.end()
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        self.table.setRowCount(len(d.top_k))
        for i, (label, p) in enumerate(d.top_k):
            self.table.setItem(i, 0, QTableWidgetItem(label))
            self.table.setItem(i, 1, QTableWidgetItem(f"{p:.2f}"))


def main() -> None:
    import sys

    app = QApplication(sys.argv)
    win = MainWindow(Config())
    win.show()
    sys.exit(app.exec())
