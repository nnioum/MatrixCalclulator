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
        self.resize(1200, 800)

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
            w.setValidator(QIntValidator(1, 5))
            w.setFixedWidth(50)
            w.textChanged.connect(self.update_size)

        size.addWidget(QLabel("Матрица A:"))
        size.addWidget(self.ra);
        size.addWidget(QLabel("x"));
        size.addWidget(self.ca)
        size.addSpacing(20)
        size.addWidget(QLabel("Матрица B:"))
        size.addWidget(self.rb);
        size.addWidget(QLabel("x"));
        size.addWidget(self.cb)
        layout.addLayout(size)

        self.A = QTableWidget(3, 3);
        self.B = QTableWidget(3, 3)
        for t in [self.A, self.B]:
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            t.setMaximumHeight(260)

        row = QHBoxLayout()
        row.addWidget(self.A, stretch=1);
        row.addWidget(QFrame(frameShape=QFrame.Shape.VLine));
        row.addWidget(self.B, stretch=1)
        layout.addLayout(row)

        btns = QHBoxLayout()
        ops = [
            ("A + B", self.mat_add), ("A - B", self.mat_sub),
            ("A × B", self.mat_mul), ("Определитель(A)", self.mat_det), ("Ранг(A)", self.mat_rank)
        ]
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
            self.out.setText(f"Результат сложения A + B:\n{A + B}")
        else:
            self.out.setText("❌ Операция невозможна: проверьте размеры (должны совпадать)")

    def mat_sub(self):
        A, B = self.M(self.A), self.M(self.B)
        if A is not None and B is not None and A.shape == B.shape:
            self.out.setText(f"Результат вычитания A - B:\n{A - B}")
        else:
            self.out.setText("❌ Операция невозможна: проверьте размеры (должны совпадать)")

    def mat_mul(self):
        A, B = self.M(self.A), self.M(self.B)
        if A is not None and B is not None and A.shape[1] == B.shape[0]:
            self.out.setText(f"Результат умножения A × B:\n{A @ B}")
        else:
            self.out.setText("❌ Операция невозможна: число столбцов A должно быть равно числу строк B")

    def mat_det(self):
        A = self.M(self.A)
        if A is not None and A.shape[0] == A.shape[1]:
            self.out.setText(f"Определитель матрицы A: {round(np.linalg.det(A), 4)}")
        else:
            self.out.setText("❌ Операция невозможна: матрица должна быть квадратной")

    def mat_rank(self):
        A = self.M(self.A)
        if A is not None: self.out.setText(f"Ранг матрицы A: {np.linalg.matrix_rank(A)}")

    # ================= ВКЛАДКА: СЛАУ =================
    def init_slau_tab(self):
        tab = QWidget()
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
        btn2 = QPushButton("Метод Крамера");
        btn2.clicked.connect(self.solve_cramer)
        btn3 = QPushButton("Метод Гаусса (Символьный)");
        btn3.clicked.connect(self.solve_gauss)
        btns.addWidget(btn2)
        btns.addWidget(btn3)
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
            self.out_s.setText("❌ Метод Крамера неприменим: определитель равен 0 (система вырождена)")
            return
        res = []
        for i in range(len(A)):
            Ai = A.copy();
            Ai[:, i] = B.flatten()
            res.append(np.linalg.det(Ai) / det_A)

        output = f"Метод Крамера:\nОпределитель Δ = {round(det_A, 4)}\n"
        for i, val in enumerate(res):
            output += f"x{i + 1} = {round(val, 4)}\n"
        self.out_s.setText(output)

    def solve_gauss(self):
        A_num, B_num = self.M(self.SA), self.M(self.SB)
        if A_num is None or B_num is None: return

        n = A_num.shape[0]
        xs = sympy.symbols(f'x1:{n + 1}')
        A_sym = sympy.Matrix(A_num)
        B_sym = sympy.Matrix(B_num)

        system = []
        for i in range(n):
            equation = sympy.Eq(sum(A_sym[i, j] * xs[j] for j in range(n)), B_sym[i])
            system.append(equation)

        try:
            solution = sympy.solve(system, xs)

            if not solution:
                self.out_s.setText("❌ Система несовместна (решений нет)")
                return

            output = "--- Результат (Зависимости переменных) ---\n"

            if isinstance(solution, dict):
                for i, x in enumerate(xs):
                    if x in solution:
                        res = sympy.simplify(solution[x])
                        output += f"{x} = {res}\n"
                    else:
                        output += f"{x} = {x} (свободный коэффициент)\n"
            else:
                output += f"Решение: {solution}"

            self.out_s.setText(output.replace('**', '^'))

        except Exception as e:
            self.out_s.setText(f"❌ Ошибка вычисления: {str(e)}")

    # ================= ВКЛАДКА: ВЕКТОРЫ =================
    def init_vector_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Векторы")
        layout = QVBoxLayout(tab)

        self.VA = QTableWidget(1, 3);
        self.VB = QTableWidget(1, 3)

        for t in [self.VA, self.VB]:
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            t.setFixedHeight(60)

        layout.addWidget(QLabel("Координаты вектора A (x, y, z):"))
        layout.addWidget(self.VA)
        layout.addWidget(QLabel("Координаты вектора B (x, y, z):"))
        layout.addWidget(self.VB)

        btns = QHBoxLayout()
        ops = [
            ("A + B", 'add'),
            ("Скалярное (dot)", 'dot'),
            ("Векторное (cross)", 'cross'),
            ("Угол между ними", 'angle')
        ]
        for n, op in ops:
            b = QPushButton(n);
            b.clicked.connect(lambda _, o=op: self.vec_calc(o));
            btns.addWidget(b)
        layout.addLayout(btns)

        self.out_v = QTextEdit();
        layout.addWidget(self.out_v)

    def vec_calc(self, op):
        self.fill_zeros(self.VA);
        self.fill_zeros(self.VB)
        try:
            a = np.array([float(self.VA.item(0, i).text().replace(',', '.')) for i in range(3)])
            b = np.array([float(self.VB.item(0, i).text().replace(',', '.')) for i in range(3)])

            if op == 'add':
                self.out_v.setText(f"Сумма векторов A + B:\n{a + b}")
            elif op == 'dot':
                self.out_v.setText(f"Скалярное произведение (A · B): {np.dot(a, b)}")
            elif op == 'cross':
                self.out_v.setText(f"Векторное произведение (A × B):\n{np.cross(a, b)}")
            elif op == 'angle':
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    self.out_v.setText("❌ Нельзя вычислить угол с нулевым вектором")
                else:
                    cos_theta = np.dot(a, b) / (norm_a * norm_b)
                    # Ограничиваем значение для точности из-за погрешностей float
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_rad = np.arccos(cos_theta)
                    angle_deg = np.degrees(angle_rad)
                    self.out_v.setText(
                        f"Угол между векторами:\nРадианы: {round(angle_rad, 4)}\nГрадусы: {round(angle_deg, 2)}°")
        except Exception as e:
            self.out_v.setText(f"❌ Ошибка данных: {str(e)}")

    # ================= ПРОЧЕЕ =================
    def init_eigen_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "Собств. числа")
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Анализ матрицы A (из первой вкладки):"))
        btn = QPushButton("Найти собственные значения и векторы");
        btn.clicked.connect(self.eigen)
        layout.addWidget(btn);
        self.eigen_out = QTextEdit();
        layout.addWidget(self.eigen_out)

    def eigen(self):
        A = self.M(self.A)
        if A is not None and A.shape[0] == A.shape[1]:
            try:
                vals, vecs = np.linalg.eig(A)
                res = "Собственные значения:\n" + str(vals)
                res += "\n\nСобственные векторы (по столбцам):\n" + str(vecs)
                self.eigen_out.setText(res)
            except Exception as e:
                self.eigen_out.setText(f"❌ Ошибка вычисления: {str(e)}")
        else:
            self.eigen_out.setText("❌ Операция невозможна: матрица должна быть квадратной")

    def init_vector_geometry_tab(self):
        tab = QWidget();
        self.tabs.addTab(tab, "Проекция")
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Проекция вектора A на вектор B (координаты из вкладки 'Векторы'):"))
        btn = QPushButton("Вычислить вектор проекции");
        btn.clicked.connect(self.proj)
        layout.addWidget(btn);
        self.geo_out = QTextEdit();
        layout.addWidget(self.geo_out)

    def proj(self):
        self.fill_zeros(self.VA);
        self.fill_zeros(self.VB)
        try:
            a = np.array([float(self.VA.item(0, i).text().replace(',', '.')) for i in range(3)])
            b = np.array([float(self.VB.item(0, i).text().replace(',', '.')) for i in range(3)])
            d = np.dot(b, b)
            if d != 0:
                projection = (np.dot(a, b) / d) * b
                self.geo_out.setText(f"Вектор проекции Pr_b(A):\n{projection}")
            else:
                self.geo_out.setText("❌ Операция невозможна: B — нулевой вектор")
        except:
            self.geo_out.setText("❌ Ошибка ввода данных")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MatrixApp()
    win.show()
    sys.exit(app.exec())