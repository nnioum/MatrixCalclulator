import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QLabel, QTextEdit, QHeaderView, QMessageBox
)
from PyQt6.QtGui import QIntValidator

import numpy as np


class CustomMatrixUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Матричный калькулятор (Ввод n x m)")
        self.resize(900, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.setup_dimension_inputs()

        self.setup_matrix_tables()

        self.setup_buttons()
        self.display_result("Ввод")

        self.update_all_matrices()

    def setup_dimension_inputs(self):
        input_layout = QHBoxLayout()

        validator = QIntValidator(1, 100)

        self.input_rows_a = QLineEdit("2")
        self.input_cols_a = QLineEdit("2")

        # Блок для Матрицы B
        self.input_rows_b = QLineEdit("2")
        self.input_cols_b = QLineEdit("2")

        for inp in [self.input_rows_a, self.input_cols_a, self.input_rows_b, self.input_cols_b]:
            inp.setValidator(validator)
            inp.setFixedWidth(40)
            inp.textChanged.connect(self.validate_and_update)

        input_layout.addWidget(QLabel("Матрица A:"))
        input_layout.addWidget(self.input_rows_a)
        input_layout.addWidget(QLabel("x"))
        input_layout.addWidget(self.input_cols_a)

        input_layout.addSpacing(40)

        input_layout.addWidget(QLabel("Матрица B:"))
        input_layout.addWidget(self.input_rows_b)
        input_layout.addWidget(QLabel("x"))
        input_layout.addWidget(self.input_cols_b)

        input_layout.addStretch()
        self.main_layout.addLayout(input_layout)

    def validate_and_update(self):
        sender = self.sender()
        text = sender.text()

        if text:
            val = int(text)
            if val > 5:
                sender.setText("5")
            elif val < 1:
                sender.setText("1")

        self.update_all_matrices()

    def update_all_matrices(self):
        try:
            ra = int(self.input_rows_a.text() or 1)
            ca = int(self.input_cols_a.text() or 1)
            self.apply_size(self.table_a, ra, ca)

            # Матрица B
            rb = int(self.input_rows_b.text() or 1)
            cb = int(self.input_cols_b.text() or 1)
            self.apply_size(self.table_b, rb, cb)
        except ValueError:
            pass

    def apply_size(self, table, rows, cols):
        table.setRowCount(rows)
        table.setColumnCount(cols)
        for i in range(rows):
            for j in range(cols):
                if not table.item(i, j):
                    table.setItem(i, j, QTableWidgetItem("0"))

    def setup_matrix_tables(self):
        layout = QHBoxLayout()
        self.table_a = QTableWidget()
        self.table_b = QTableWidget()
        layout.addWidget(self.table_a)
        layout.addWidget(self.table_b)
        self.main_layout.addLayout(layout)

    def setup_buttons(self):
        layout = QHBoxLayout()
        # ["A+B", "A-B", "A*B", "Det(A)", "Rank(A)", "СЛАУ (Гаусс)"]
        commands = {
            "A+B": self.get_summ,
            "A-B": self.get_sub,
            "A*B": self.get_mul,
            "Transpose(A)": self.get_transpose,
            "Det(A)": self.get_det,
            "Rank(A)": self.get_rank
        }
        for name, func in commands.items():
            btn = QPushButton(name)
            btn.clicked.connect(func)
            layout.addWidget(btn)
        self.main_layout.addLayout(layout)

    def display_result(self, data, is_matrix=False):
        if hasattr(self, 'result_widget'):
            self.main_layout.removeWidget(self.result_widget)
            self.result_widget.deleteLater()

        if is_matrix:
            rows, cols = data.shape
            self.result_widget = QTableWidget(rows, cols)
            self.result_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

            for i in range(rows):
                for j in range(cols):
                    val = str(round(data[i, j], 2))
                    self.result_widget.setItem(i, j, QTableWidgetItem(val))

            self.result_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        else:
            self.result_widget = QTextEdit()
            self.result_widget.setPlainText(str(data))
            self.result_widget.setReadOnly(True)
            self.result_widget.setMaximumHeight(100)

        self.main_layout.addWidget(self.result_widget)

    def get_matrix(self, table):
        try:
            n = table.rowCount()
            m = table.columnCount()
            data = []
            for i in range(n):
                row = []
                for j in range(m):
                    item = table.item(i, j)
                    txt = item.text().strip() if item else ""
                    row.append(float(txt) if txt else 0.0)
                data.append(row)
            return np.array(data)
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Вводите только числа! Проверьте, нет ли букв или лишних точек.")
            return None

    def get_summ(self):
        matrix_a = self.get_matrix(self.table_a)
        matrix_b = self.get_matrix(self.table_b)
        if matrix_a is not None and matrix_b is not None:
            if matrix_a.shape == matrix_b.shape:
                matrix_sum = matrix_a + matrix_b
                self.display_result(matrix_sum, True)
            else:
                QMessageBox.warning(self, "Ошибка", "Для сложения матрицы должны быть одинакового размера!")

    def get_sub(self):
        matrix_a = self.get_matrix(self.table_a)
        matrix_b = self.get_matrix(self.table_b)
        if matrix_a is not None and matrix_b is not None:
            if matrix_a.shape == matrix_b.shape:
                matrix_sum = matrix_a - matrix_b
                self.display_result(matrix_sum, True)
            else:
                QMessageBox.warning(self, "Ошибка", "Для вычитания матрицы должны быть одинакового размера!")

    def get_mul(self):
        matrix_a = self.get_matrix(self.table_a)
        matrix_b = self.get_matrix(self.table_b)
        if matrix_a is not None and matrix_b is not None:
            if matrix_a.shape[1] == matrix_b.shape[0]:
                matrix_sum = np.matmul(matrix_a, matrix_b)
                self.display_result(matrix_sum, True)
            else:
                QMessageBox.warning(self, "Ошибка", "Для умножения количество столбцов первой матрицы должны быть равны количеству строк второй матрицы!")

    def get_transpose(self):
        matrix_a = self.get_matrix(self.table_a)
        if matrix_a is not None:
            matrix = matrix_a.T
            self.display_result(matrix, True)

    def get_det(self):
        matrix_a = self.get_matrix(self.table_a)
        if matrix_a is not None:
            matrix = np.linalg.det(matrix_a)
            self.display_result(matrix)


    def get_rank(self):
        matrix_a = self.get_matrix(self.table_a)
        if matrix_a is not None:
            matrix = np.linalg.matrix_rank(matrix_a)
            self.display_result(matrix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CustomMatrixUI()
    window.show()
    sys.exit(app.exec())
