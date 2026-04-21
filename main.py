import sys
import numpy as np
import sympy

from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit,
    QTabWidget, QLineEdit, QLabel, QFrame, QInputDialog,
    QHeaderView, QSizePolicy
)


class MatrixApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Матричный калькулятор PRO")
        self.resize(1200, 900)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.init_matrix_tab()
        self.init_slau_tab()
        self.init_vector_tab()
        self.init_eigen_tab()
        self.init_vector_geometry_tab()

    # ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
    def fill_zeros(self, table):
        for i in range(table.rowCount()):
            for j in range(table.columnCount()):
                item = table.item(i, j)
                if item is None or item.text().strip() == "":
                    table.setItem(i, j, QTableWidgetItem("0"))

    def safe_int(self, text):
        try:
            v = int(text)
            return max(1, min(10, v))
        except:
            return 3

    def M(self, t):
        self.fill_zeros(t)
        try:
            return np.array([
                [float(t.item(i, j).text().replace(',', '.')) for j in range(t.columnCount())]
                for i in range(t.rowCount())
            ])
        except ValueError:
            return None

    # ================= ВКЛАДКА: МАТРИЦЫ =================
    def init_matrix_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Матрицы")
        layout = QVBoxLayout(tab)

        size = QHBoxLayout()
        self.ra = QLineEdit("3");
        self.ca = QLineEdit("3")
        self.rb = QLineEdit("3");
        self.cb = QLineEdit("3")

        for w in [self.ra, self.ca, self.rb, self.cb]:
            w.setValidator(QIntValidator(1, 10))
            w.setFixedWidth(40)
            w.textChanged.connect(self.update_size)

        size.addWidget(QLabel("Матрица A:"));
        size.addWidget(self.ra);
        size.addWidget(QLabel("x"));
        size.addWidget(self.ca)
        size.addSpacing(40)
        size.addWidget(QLabel("Матрица B:"));
        size.addWidget(self.rb);
        size.addWidget(QLabel("x"));
        size.addWidget(self.cb)
        layout.addLayout(size)

        self.A = QTableWidget(3, 3);
        self.B = QTableWidget(3, 3)
        for t in [self.A, self.B]:
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            t.setMaximumHeight(250)

        row = QHBoxLayout()
        row.addWidget(self.A, stretch=1);
        row.addWidget(QFrame(frameShape=QFrame.Shape.VLine));
        row.addWidget(self.B, stretch=1)
        layout.addLayout(row)

        btns = QVBoxLayout()
        row1 = QHBoxLayout()
        ops1 = [("A + B", self.mat_add), ("A - B", self.mat_sub), ("A × B", self.mat_mul)]
        for name, func in ops1:
            btn = QPushButton(name);
            btn.clicked.connect(func);
            row1.addWidget(btn)

        row2 = QHBoxLayout()
        ops2 = [("A × n", self.mat_mul_num), ("det(A)", self.mat_det), ("Ранг(A)", self.mat_rank),
                ("Трансп.(A)", self.mat_transp)]
        for name, func in ops2:
            btn = QPushButton(name);
            btn.clicked.connect(func);
            row2.addWidget(btn)

        btns.addLayout(row1)
        btns.addLayout(row2)
        layout.addLayout(btns)

        self.out = QTextEdit();
        self.out.setReadOnly(True)
        layout.addWidget(self.out)

    def update_size(self):
        self.A.setRowCount(self.safe_int(self.ra.text()))
        self.A.setColumnCount(self.safe_int(self.ca.text()))
        self.B.setRowCount(self.safe_int(self.rb.text()))
        self.B.setColumnCount(self.safe_int(self.cb.text()))

    def mat_add(self):
        A, B = self.M(self.A), self.M(self.B)
        if A is not None and B is not None and A.shape == B.shape:
            self.out.setText(f"Результат A + B:\n{A + B}")
        else:
            self.out.setText("❌ Ошибка: размеры матриц должны совпадать")

    def mat_sub(self):
        A, B = self.M(self.A), self.M(self.B)
        if A is not None and B is not None and A.shape == B.shape:
            self.out.setText(f"Результат A - B:\n{A - B}")
        else:
            self.out.setText("❌ Ошибка: размеры матриц должны совпадать")

    def mat_mul(self):
        A, B = self.M(self.A), self.M(self.B)
        if A is not None and B is not None and A.shape[1] == B.shape[0]:
            self.out.setText(f"Результат A × B:\n{A @ B}")
        else:
            self.out.setText("❌ Ошибка: количество столбцов A должно быть равно строкам B")

    def mat_mul_num(self):
        A = self.M(self.A)
        if A is not None:
            n, ok = QInputDialog.getDouble(self, "Умножение", "Введите число n:", 1.0)
            if ok: self.out.setText(f"Результат A × {n}:\n{A * n}")

    def mat_transp(self):
        A = self.M(self.A)
        if A is not None: self.out.setText(f"Транспонированная матрица A:\n{A.T}")

    def mat_det(self):
        A = self.M(self.A)
        if A is not None and A.shape[0] == A.shape[1]:
            self.out.setText(f"Определитель A: {round(np.linalg.det(A), 4)}")
        else:
            self.out.setText("❌ Ошибка: матрица должна быть квадратной")

    def mat_rank(self):
        A = self.M(self.A)
        if A is not None: self.out.setText(f"Ранг A: {np.linalg.matrix_rank(A)}")

    # ================= ВКЛАДКА: СЛАУ =================
    def init_slau_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "СЛАУ")
        layout = QVBoxLayout(tab)

        header = QHBoxLayout()
        self.n_slau = QLineEdit("3");
        self.n_slau.setFixedWidth(50)
        self.n_slau.textChanged.connect(self.update_slau)
        header.addWidget(QLabel("Количество неизвестных n:"));
        header.addWidget(self.n_slau);
        header.addStretch()
        layout.addLayout(header)

        self.SA = QTableWidget(3, 3);
        self.SB = QTableWidget(3, 1)
        row = QHBoxLayout();
        row.addWidget(self.SA, 3);
        row.addWidget(QLabel("="));
        row.addWidget(self.SB, 1)
        layout.addLayout(row)

        btns = QHBoxLayout()
        b_mat = QPushButton("Матричный метод");
        b_mat.clicked.connect(self.solve_matrix_method)
        b_cr = QPushButton("Метод Крамера");
        b_cr.clicked.connect(self.solve_cramer)
        b_gs = QPushButton("Метод Гаусса");
        b_gs.clicked.connect(self.solve_gauss)
        btns.addWidget(b_mat);
        btns.addWidget(b_cr);
        btns.addWidget(b_gs)
        layout.addLayout(btns)

        self.out_s = QTextEdit();
        self.out_s.setReadOnly(True)
        layout.addWidget(self.out_s)

    def update_slau(self):
        n = self.safe_int(self.n_slau.text())
        self.SA.setRowCount(n);
        self.SA.setColumnCount(n)
        self.SB.setRowCount(n);
        self.SB.setColumnCount(1)

    def solve_matrix_method(self):
        A, B = self.M(self.SA), self.M(self.SB)
        if A is None or B is None: return
        try:
            res = np.linalg.inv(A) @ B
            self.out_s.setText("Матричный метод (X = A⁻¹ * B):\n" + "\n".join(
                [f"x{i + 1} = {round(v[0], 4)}" for i, v in enumerate(res)]))
        except np.linalg.LinAlgError:
            self.out_s.setText("❌ Ошибка: матрица вырождена (det=0), обратной матрицы не существует")

    def solve_cramer(self):
        A, B = self.M(self.SA), self.M(self.SB)
        if A is None or B is None: return
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-9:
            self.out_s.setText("❌ Ошибка: det = 0, метод Крамера неприменим")
            return
        res = []
        for i in range(len(A)):
            Ai = A.copy()
            Ai[:, i] = B[:, 0]
            res.append(np.linalg.det(Ai) / det_A)
        self.out_s.setText("Метод Крамера:\n" + "\n".join([f"x{i + 1} = {round(v, 4)}" for i, v in enumerate(res)]))

    def solve_gauss(self):
        A_num, B_num = self.M(self.SA), self.M(self.SB)
        if A_num is None or B_num is None: return
        n = A_num.shape[0]
        xs = sympy.symbols(f'x1:{n + 1}')
        system = [sympy.Eq(sum(A_num[i, j] * xs[j] for j in range(n)), B_num[i, 0]) for i in range(n)]
        try:
            sol = sympy.solve(system, xs)
            if not sol: self.out_s.setText("❌ Решений нет"); return
            out = "--- Метод Гаусса (Символьный) ---\n"
            if isinstance(sol, dict):
                for x in xs: out += f"{x} = {sympy.simplify(sol.get(x, x))}\n"
            else:
                out += f"Решение: {sol}"
            self.out_s.setText(out.replace('**', '^'))
        except Exception as e:
            self.out_s.setText(f"❌ Ошибка: {e}")

    # ================= ВКЛАДКА: ВЕКТОРЫ =================
    def init_vector_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "Векторы")
        layout = QVBoxLayout(tab)
        self.VA = QTableWidget(1, 3);
        self.VB = QTableWidget(1, 3);
        self.VC = QTableWidget(1, 3)
        for t in [self.VA, self.VB, self.VC]:
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            t.setFixedHeight(60)

        layout.addWidget(QLabel("Вектор A (x, y, z):"));
        layout.addWidget(self.VA)
        layout.addWidget(QLabel("Вектор B (x, y, z):"));
        layout.addWidget(self.VB)
        layout.addWidget(QLabel("Вектор C (для смешанного):"));
        layout.addWidget(self.VC)

        btns = QVBoxLayout()
        row1 = QHBoxLayout()
        for n, op in [("A + B", 'add'), ("A - B", 'sub'), ("A × n", 'mul_n')]:
            btn = QPushButton(n);
            btn.clicked.connect(lambda _, o=op: self.vec_calc(o));
            row1.addWidget(btn)

        row2 = QHBoxLayout()
        for n, op in [("Скалярное (A·B)", 'dot'), ("Векторное (AxB)", 'cross'), ("Смешанное (ABC)", 'mixed')]:
            btn = QPushButton(n);
            btn.clicked.connect(lambda _, o=op: self.vec_calc(o));
            row2.addWidget(btn)

        btns.addLayout(row1);
        btns.addLayout(row2)
        layout.addLayout(btns)
        self.out_v = QTextEdit();
        self.out_v.setReadOnly(True)
        layout.addWidget(self.out_v)

    def vec_calc(self, op):
        try:
            a = np.array(
                [float(self.VA.item(0, i).text().replace(',', '.')) if self.VA.item(0, i) else 0.0 for i in range(3)])
            b = np.array(
                [float(self.VB.item(0, i).text().replace(',', '.')) if self.VB.item(0, i) else 0.0 for i in range(3)])

            if op == 'add':
                self.out_v.setText(f"A + B = {a + b}")
            elif op == 'sub':
                self.out_v.setText(f"A - B = {a - b}")
            elif op == 'mul_n':
                n, ok = QInputDialog.getDouble(self, "Вектор", "Введите число n:", 1.0)
                if ok: self.out_v.setText(f"A × {n} = {a * n}")
            elif op == 'dot':
                self.out_v.setText(f"Скалярное произведение A · B = {np.dot(a, b)}")
            elif op == 'cross':
                self.out_v.setText(f"Векторное произведение A × B = {np.cross(a, b)}")
            elif op == 'mixed':
                self.fill_zeros(self.VC)
                c = np.array([float(self.VC.item(0, i).text().replace(',', '.')) for i in range(3)])
                res = np.dot(a, np.cross(b, c))
                self.out_v.setText(f"Смешанное произведение (ABC) = {round(res, 4)}")
        except Exception as e:
            self.out_v.setText(f"❌ Ошибка: {e}")

    # ================= ПРОЧЕЕ =================
    def init_eigen_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "Собств. числа")
        layout = QVBoxLayout(tab)
        btn = QPushButton("Найти собств. значения и векторы матрицы A");
        btn.clicked.connect(self.eigen)
        layout.addWidget(btn)
        self.eigen_out = QTextEdit();
        self.eigen_out.setReadOnly(True)
        layout.addWidget(self.eigen_out)

    def eigen(self):
        A = self.M(self.A)
        if A is not None and A.shape[0] == A.shape[1]:
            v, vec = np.linalg.eig(A)
            self.eigen_out.setText(f"Собственные числа:\n{v}\n\nСобственные векторы (по столбцам):\n{vec}")
        else:
            self.eigen_out.setText("❌ Ошибка: нужна квадратная матрица A")

    def init_vector_geometry_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "Проекция")
        layout = QVBoxLayout(tab)
        btn = QPushButton("Проекция вектора A на вектор B");
        btn.clicked.connect(self.proj)
        layout.addWidget(btn)
        self.geo_out = QTextEdit();
        self.geo_out.setReadOnly(True)
        layout.addWidget(self.geo_out)

    def proj(self):
        try:
            a = np.array(
                [float(self.VA.item(0, i).text().replace(',', '.')) if self.VA.item(0, i) else 0.0 for i in range(3)])
            b = np.array(
                [float(self.VB.item(0, i).text().replace(',', '.')) if self.VB.item(0, i) else 0.0 for i in range(3)])
            if np.dot(b, b) != 0:
                p = (np.dot(a, b) / np.dot(b, b)) * b
                self.geo_out.setText(f"Вектор проекции A на B:\n{p}")
            else:
                self.geo_out.setText("❌ Ошибка: Вектор B не может быть нулевым")
        except Exception as e:
            self.geo_out.setText(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MatrixApp()
    win.show()
    sys.exit(app.exec())