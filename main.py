import sys
import numpy as np
import sympy

from PyQt6.QtGui import QIntValidator
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
        except:
            return 1
        return max(1, min(5, v))

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
            w.setValidator(QIntValidator(1, 5));
            w.setFixedWidth(50)
            w.textChanged.connect(self.update_size)

        size.addWidget(QLabel("Матрица A:"));
        size.addWidget(self.ra);
        size.addWidget(QLabel("x"));
        size.addWidget(self.ca)
        size.addSpacing(20)
        size.addWidget(QLabel("Матрица B:"));
        size.addWidget(self.rb);
        size.addWidget(QLabel("x"));
        size.addWidget(self.cb)
        layout.addLayout(size)

        self.A = QTableWidget(3, 3);
        self.B = QTableWidget(3, 3)
        for t in [self.A, self.B]:
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            t.setMaximumHeight(200)

        row = QHBoxLayout()
        row.addWidget(self.A, stretch=1);
        row.addWidget(QFrame(frameShape=QFrame.Shape.VLine));
        row.addWidget(self.B, stretch=1)
        layout.addLayout(row)

        btns = QHBoxLayout()
        ops = [("A + B", self.mat_add), ("A - B", self.mat_sub), ("A × B", self.mat_mul), ("det(A)", self.mat_det),
               ("Ранг(A)", self.mat_rank)]
        for name, func in ops:
            btn = QPushButton(name);
            btn.clicked.connect(func);
            btns.addWidget(btn)
        layout.addLayout(btns)

        self.out = QTextEdit();
        self.out.setMinimumHeight(120);
        layout.addWidget(self.out)

    def update_size(self):
        self.A.setRowCount(self.safe_int(self.ra.text()));
        self.A.setColumnCount(self.safe_int(self.ca.text()))
        self.B.setRowCount(self.safe_int(self.rb.text()));
        self.B.setColumnCount(self.safe_int(self.cb.text()))

    def mat_add(self):
        A, B = self.M(self.A), self.M(self.B)
        if A is not None and B is not None and A.shape == B.shape:
            self.out.setText(f"Результат A + B:\n{A + B}")
        else:
            self.out.setText("❌ Ошибка: размеры должны совпадать")

    def mat_sub(self):
        A, B = self.M(self.A), self.M(self.B)
        if A is not None and B is not None and A.shape == B.shape:
            self.out.setText(f"Результат A - B:\n{A - B}")
        else:
            self.out.setText("❌ Ошибка: размеры должны совпадать")

    def mat_mul(self):
        A, B = self.M(self.A), self.M(self.B)
        if A is not None and B is not None and A.shape[1] == B.shape[0]:
            self.out.setText(f"Результат A × B:\n{A @ B}")
        else:
            self.out.setText("❌ Ошибка: столбцы A != строки B")

    def mat_det(self):
        A = self.M(self.A)
        if A is not None and A.shape[0] == A.shape[1]:
            self.out.setText(f"Определитель A: {round(np.linalg.det(A), 4)}")
        else:
            self.out.setText("❌ Ошибка: матрица не квадратная")

    def mat_rank(self):
        A = self.M(self.A)
        if A is not None: self.out.setText(f"Ранг A: {np.linalg.matrix_rank(A)}")

    # ================= ВКЛАДКА: СЛАУ =================
    def init_slau_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "СЛАУ")
        layout = QVBoxLayout(tab)
        self.n_slau = QLineEdit("3");
        self.n_slau.setFixedWidth(50);
        self.n_slau.textChanged.connect(self.update_slau)
        layout.addWidget(QLabel("Количество неизвестных n:"));
        layout.addWidget(self.n_slau)
        self.SA = QTableWidget(3, 3);
        self.SB = QTableWidget(3, 1)
        row = QHBoxLayout();
        row.addWidget(self.SA);
        row.addWidget(QLabel("="));
        row.addWidget(self.SB)
        layout.addLayout(row)
        btns = QHBoxLayout()
        btn_cr = QPushButton("Метод Крамера");
        btn_cr.clicked.connect(self.solve_cramer)
        btn_gs = QPushButton("Метод Гаусса (Символьный)");
        btn_gs.clicked.connect(self.solve_gauss)
        btns.addWidget(btn_cr);
        btns.addWidget(btn_gs)
        layout.addLayout(btns)
        self.out_s = QTextEdit();
        layout.addWidget(self.out_s)

    def update_slau(self):
        n = self.safe_int(self.n_slau.text())
        self.SA.setRowCount(n);
        self.SA.setColumnCount(n);
        self.SB.setRowCount(n)

    def solve_cramer(self):
        A, B = self.M(self.SA), self.M(self.SB)
        if A is None or B is None: return
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-9:
            self.out_s.setText("❌ Ошибка: det = 0")
            return
        res = [np.linalg.det(np.column_stack([A[:, :i], B, A[:, i + 1:]])) / det_A for i in range(len(A))]
        self.out_s.setText("Метод Крамера:\n" + "\n".join([f"x{i + 1} = {round(v, 4)}" for i, v in enumerate(res)]))

    def solve_gauss(self):
        A_num, B_num = self.M(self.SA), self.M(self.SB)
        if A_num is None or B_num is None: return
        n = A_num.shape[0];
        xs = sympy.symbols(f'x1:{n + 1}')
        system = [sympy.Eq(sum(A_num[i, j] * xs[j] for j in range(n)), B_num[i]) for i in range(n)]
        try:
            sol = sympy.solve(system, xs)
            if not sol: self.out_s.setText("❌ Решений нет"); return
            out = "--- Зависимости переменных ---\n"
            if isinstance(sol, dict):
                for x in xs: out += f"{x} = {sympy.simplify(sol[x]) if x in sol else str(x) + ' (свободный)'}\n"
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
            t.setFixedHeight(50)

        layout.addWidget(QLabel("Вектор A:"));
        layout.addWidget(self.VA)
        layout.addWidget(QLabel("Вектор B:"));
        layout.addWidget(self.VB)
        layout.addWidget(QLabel("Вектор C (для смешанного произв.):"));
        layout.addWidget(self.VC)

        btns = QHBoxLayout()
        ops = [("A + B", 'add'), ("Скалярное (A·B)", 'dot'), ("Векторное (AxB)", 'cross'), ("Смешанное (ABC)", 'mixed'),
               ("Угол (A,B)", 'angle')]
        for n, op in ops:
            b = QPushButton(n);
            b.clicked.connect(lambda _, o=op: self.vec_calc(o));
            btns.addWidget(b)
        layout.addLayout(btns)
        self.out_v = QTextEdit();
        layout.addWidget(self.out_v)

    def vec_calc(self, op):
        self.fill_zeros(self.VA);
        self.fill_zeros(self.VB);
        self.fill_zeros(self.VC)
        try:
            a = np.array([float(self.VA.item(0, i).text().replace(',', '.')) for i in range(3)])
            b = np.array([float(self.VB.item(0, i).text().replace(',', '.')) for i in range(3)])
            c = np.array([float(self.VC.item(0, i).text().replace(',', '.')) for i in range(3)])

            if op == 'add':
                self.out_v.setText(f"A + B = {a + b}")
            elif op == 'dot':
                self.out_v.setText(f"A · B = {np.dot(a, b)}")
            elif op == 'cross':
                self.out_v.setText(f"A × B = {np.cross(a, b)}")
            elif op == 'mixed':
                # Смешанное произведение: det([a, b, c])
                res = np.dot(a, np.cross(b, c))
                self.out_v.setText(f"Смешанное произведение (ABC) = {round(res, 4)}")
            elif op == 'angle':
                cos_t = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                angle = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
                self.out_v.setText(f"Угол: {round(angle, 2)}°")
        except Exception as e:
            self.out_v.setText(f"❌ Ошибка: {e}")

    # ================= ПРОЧЕЕ =================
    def init_eigen_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "Собств. числа")
        layout = QVBoxLayout(tab);
        btn = QPushButton("Найти собств. значения A");
        btn.clicked.connect(self.eigen)
        layout.addWidget(btn);
        self.eigen_out = QTextEdit();
        layout.addWidget(self.eigen_out)

    def eigen(self):
        A = self.M(self.A)
        if A is not None and A.shape[0] == A.shape[1]:
            v, vec = np.linalg.eig(A);
            self.eigen_out.setText(f"Числа: {v}\n\nВекторы:\n{vec}")
        else:
            self.eigen_out.setText("❌ Ошибка: нужна квадратная матрица A")

    def init_vector_geometry_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "Проекция")
        layout = QVBoxLayout(tab);
        btn = QPushButton("Проекция A на B");
        btn.clicked.connect(self.proj)
        layout.addWidget(btn);
        self.geo_out = QTextEdit();
        layout.addWidget(self.geo_out)

    def proj(self):
        self.fill_zeros(self.VA);
        self.fill_zeros(self.VB)
        a, b = self.M(self.VA)[0], self.M(self.VB)[0]
        if np.dot(b, b) != 0:
            self.geo_out.setText(f"Проекция: {(np.dot(a, b) / np.dot(b, b)) * b}")
        else:
            self.geo_out.setText("❌ Ошибка: B=0")


if __name__ == "__main__":
    app = QApplication(sys.argv);
    win = MatrixApp();
    win.show();
    sys.exit(app.exec())