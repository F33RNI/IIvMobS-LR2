"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""
import os
import sys
import traceback

import numpy as np
import sympy
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

# Put your GUI file here
GUI_FILE = "gui.ui"
GUI_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), GUI_FILE))

# "fusion" or "windows"
GUI_STYLE = "fusion"


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()

        # 3D lists for storing calculated population's points instead af pictures of the plots as files =)
        # (x, y) * points * populations
        # ex.: point_x = self.population_points_bad[population_index][point_n][0]
        self.points_bad = []
        self.points_good = []

        # Numpy arrays for storing function (for plotting)
        self.x_values = np.empty((0,), dtype=np.float32)
        self.y_values = np.empty((0,), dtype=np.float32)

        # Load GUI from file
        uic.loadUi(GUI_FILE, self)

        # Initialize matplotlib
        self.plt_view = FigureCanvasQTAgg(Figure(tight_layout=True))
        self.plt_axes = self.plt_view.figure.subplots()
        self.plt_axes.grid(True, which="both")
        plt_toolbar = NavigationToolbar2QT(self.plt_view, self)
        v_box_layout = QVBoxLayout()
        v_box_layout.addWidget(plt_toolbar)
        v_box_layout.addWidget(self.plt_view)
        self.widget.setLayout(v_box_layout)

        # Connect button
        self.pushButton.clicked.connect(self.start)

        # Connect slider
        self.horizontalSlider.valueChanged.connect(self.change_view)

        # Show GUI
        self.show()

    def start(self) -> None:
        """
        Start button callback. Main program entry
        :return:
        """
        try:
            # Parse function from GUI
            function = sympy.lambdify(sympy.Symbol("x"), self.lineEdit.text())

            # Retrieve other data from elements
            border_left = self.doubleSpinBox.value()
            border_right = self.doubleSpinBox_2.value()
            plot_margins = self.doubleSpinBox_3.value()
            optimum = self.comboBox.currentIndex()
            population_count = self.spinBox.value()
            mutation_intensity = self.doubleSpinBox_4.value()
            mutation_frequency = self.doubleSpinBox_5.value()
            agents_number = self.spinBox_2.value()

            # Set population count as maximum for the slider and reset it
            self.horizontalSlider.setMaximum(population_count - 1)
            self.horizontalSlider.setValue(0)

            # Clear population points
            self.points_bad.clear()
            self.points_good.clear()

            # Calculate function (for plotting)
            self.x_values = np.arange(border_left - plot_margins, border_right + plot_margins, 0.01, dtype=np.float32)
            self.y_values = function(self.x_values)

            # Initialize agents and append them into list
            agents = []
            for i in range(agents_number):
                agent_x = (border_right - border_left) * np.random.rand() + border_left
                agent = Agent(agent_x)
                agent.calculate(function)
                agents.append(agent)

            for population_index in range(population_count):
                # Initialize lists for storing current population's points
                population_points_bad = []
                population_points_good = []

                # Find bad points (maximums or minimums) using half of agents
                # (delete half of the agents, the worst ones)
                for _ in range(agents_number // 2):
                    # Index of rejected agent
                    index = 0

                    # 0 - Max (selected item's index from comboBox)
                    # Find minimum (because minimum is the worst)
                    if optimum == 0:
                        minimum = np.inf
                        for agent_index in range(len(agents)):
                            agent = agents[agent_index]
                            if agent.r_y() < minimum:
                                minimum = agent.r_y()
                                index = agent_index

                    # 1 - Min (selected item's index from comboBox)
                    # Find maximum (because maximum is the worst)
                    else:
                        maximum = -np.inf
                        for agent_index in range(len(agents)):
                            agent = agents[agent_index]
                            if agent.r_y() > maximum:
                                maximum = agent.r_y()
                                index = agent_index

                    # Save it as bad point
                    population_points_bad.append(agents[index].as_point())

                    # Delete this agent because it's bad
                    del agents[index]

                # Consider the remaining agents (half of them) as good ones
                for agent in agents:
                    population_points_good.append(agent.as_point())

                # Save current population points
                self.points_bad.append(population_points_bad)
                self.points_good.append(population_points_good)

                # Duplicate remaining agents (good ones)
                for agent_index_half in range(len(agents)):
                    agents.append(Agent(agents[agent_index_half].r_x()))

                # Mutate them and calculate Y (function) for mutated X
                for agent in agents:
                    agent.mutate(mutation_intensity, mutation_frequency)
                    agent.calculate(function)

            # Show first plot
            self.plot_population(0)

        # Log error in main function if it occurs
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText(str(e))
            msg.setWindowTitle("Error")
            msg.exec_()
            traceback.print_exc()

    def change_view(self) -> None:
        """
        Slider callback. Changes label text and plots selected population
        :return:
        """
        population_index_to_plot = self.horizontalSlider.value()
        self.label_11.setText(str(population_index_to_plot + 1))
        if self.x_values.shape[0] > population_index_to_plot and self.y_values.shape[0] > population_index_to_plot:
            self.plot_population(population_index_to_plot)

    def plot_population(self, population_index: int) -> None:
        """
        Plots current function and bad and good points for certain population index
        :param population_index: index of points to plot
        :return:
        """
        # Clear current plot
        self.plt_axes.clear()

        # Enable grid
        self.plt_axes.grid(True, which="both")

        # Plot function
        self.plt_axes.plot(self.x_values, self.y_values)

        # Plot bad points using red color
        for point in self.points_bad[population_index]:
            self.plt_axes.scatter(point[0], point[1], c="r")

        # Plot good points using green color
        for point in self.points_good[population_index]:
            self.plt_axes.scatter(point[0], point[1], c="g")

        # Update plot
        self.plt_view.draw()


class Agent:
    def __init__(self, x) -> None:
        # Initialize private variables
        self._x = x
        self._y = None

    def mutate(self, intensity: float, frequency: float) -> None:
        """
        Calculates one mutation step
        :param intensity: Intensity of mutation
        :param frequency: Frequency of mutation
        :return:
        """
        if np.random.rand() <= frequency:
            if np.random.rand() > .5:
                self._x += intensity
            else:
                self._x -= intensity

    def calculate(self, function) -> None:
        """
        Applies function to calculate y from x
        :param function: parsed function
        :return:
        """
        self._y = function(self._x)

    def r_x(self) -> float:
        """
        Returns private x variable value
        :return:
        """
        return self._x

    def r_y(self) -> float | None:
        """
        Returns private y variable value
        :return:
        """
        return self._y

    def as_point(self) -> np.ndarray:
        """
        Returns x and y
        :return: x and y as numpy array
        """
        return np.asarray([self._x, self._y], dtype=np.float32)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # Start GUI
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle(GUI_STYLE)
    win = Window()
    sys.exit(app.exec_())
