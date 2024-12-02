import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QMainWindow, QGridLayout, QWidget, QApplication, QLabel
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from PyQt6 import QtGui, QtCore


class RectItem(pg.GraphicsObject):
    def __init__(self, rect, parent=None):
        super().__init__(parent)
        self._rect = rect
        self.picture = QtGui.QPicture()
        self._generate_picture()

    @property
    def rect(self):
        return self._rect

    def _generate_picture(self):
        painter = QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen("r"))
        #painter.setBrush(pg.mkBrush("g"))
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

class Color(QWidget):
    def __init__(self, color, fill=True):
        super(Color, self).__init__()
        self.setAutoFillBackground(fill)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)

class PyQtRenderer(QMainWindow):
    def __init__(self):
        super(PyQtRenderer, self).__init__()
        # Main Window Parameters
        self.win_width = 1500
        self.win_height = 750
        self.col_min_width = 500
        self.max_steps = 10

        # Start Main Window Init
        self.setFixedWidth(self.win_width)
        self.setFixedHeight(self.win_height)
        self.main_panel = QGridLayout()
        self.pen_color = 'w'
        
        # MUT Panel Init (Left Panel)
        self.mut_label = QLabel('<b>Model Under Test (MUT)<\b>')
        self.mut_label.setFixedHeight(20)
        self.mut_label.setFixedWidth(self.col_min_width)
        self.mut_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.observation_widget, self.observationPlotItem = self.__initialize_observation_view()
        self.tpr_widget = self.__initialize_cummulative_tpr_view()
        self.fpr_widget = self.__initialize_cummulative_fpr_view()

        self.main_panel.addWidget(self.mut_label,                           0, 0, 1, 1)
        self.main_panel.addWidget(self.observation_widget,                  1, 0, 1, 1)
        self.main_panel.addWidget(self.tpr_widget,                          2, 0, 1, 1)
        self.main_panel.addWidget(self.fpr_widget,                          3, 0, 1, 1)

        
        # MASTIFF Panel Init (Center Panel)
        self.mastiff_label = QLabel('<b>The MASTIFF Gymnasium<\b>')
        self.mastiff_label.setFixedHeight(20)
        self.mastiff_label.setFixedWidth(self.col_min_width)
        self.mastiff_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.mastiff_panel = QGridLayout()
        #self.mastiff_panel.setColumnMinimumWidth(0, self.col_min_width)
        self.spectrum_widget, self.spectrum_image_item, self.spectrumPlotItem = self.__initialize_spectrum_view()
        self.mastiff_panel.addWidget(self.spectrum_widget,                              0, 0, 1, 1)
        self.logo_panel = QGridLayout()        
        self.logo_widget = self.__initialize_logo_view()
        self.logo_panel.addWidget(Color('white', fill=False),                                       0, 0, 1, 1)
        self.logo_panel.addWidget(self.logo_widget,                                                 0, 1, 1, 1)
        self.logo_panel.addWidget(Color('white', fill=False),                                       0, 2, 1, 1)
        self.mastiff_panel.addLayout(self.logo_panel,                                   1, 0, 1, 1)
        #self.mastiff_panel.addWidget(Color('white', fill=False),                                    2, 0, 1, 1)

        self.main_panel.addWidget(self.mastiff_label,                       0, 1, 1, 1)
        self.main_panel.addLayout(self.mastiff_panel,                       1, 1, 2, 1)

        # Adversarial Agent Panel Init (Right Panel)
        self.adversarial_agent_label = QLabel('<b>Adversarial Agent<\b>')
        self.adversarial_agent_label.setFixedHeight(20)
        self.adversarial_agent_label.setFixedWidth(self.col_min_width)
        self.adversarial_agent_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.bounding_box_widget, self.bbPlotItem = self.__initialize_bounding_box_view()
        self.power_widget = self.__initialize_power_view()
        self.reward_widget = self.__initialize_cummulative_reward_view()

        self.main_panel.addWidget(self.adversarial_agent_label,             0, 2, 1, 1)
        self.main_panel.addWidget(self.bounding_box_widget,                 1, 2, 1, 1) 
        self.main_panel.addWidget(self.power_widget,                        2, 2, 1, 1)  
        self.main_panel.addWidget(self.reward_widget,                       3, 2, 1, 1)    
        #self.main_panel.addWidget(Color('red'),                             1, 2, 3, 1)
       
        # Finish Main Window Init
        self.widget = QWidget()
        self.widget.setLayout(self.main_panel)
        self.setCentralWidget(self.widget)     

        self.bb_items = []
        self.mut_items = []

        self.step = 1

    def render(self, info):           
        self.spectrum_image_item.setImage(np.flipud(np.rot90(info['spectrum'],k=1)))

        #self.reset_bb(self.bb_items)
        #self.reset_mut(self.mut_items)

        # plot bounding boxes (ground truth)
        self.bb_items = []
        if info['bounding_boxes'] != []:
            for box in info['bounding_boxes']:
                rect_item = RectItem(QtCore.QRectF(box[1]*640, box[3]*640+(self.step-1)*640, (box[2]-box[1])*640, (box[4]-box[3])*640))
                self.bb_items.append(rect_item)
                self.bbPlotItem.addItem(rect_item)

        # plot bounding boxes (MUT)
        self.mut_items = []
        if info['observation'] != []:
            for box in info['observation']:
                box = box.tolist()
                print(box)
                rect_item =  RectItem(QtCore.QRectF(box[0],box[1]+(self.step-1)*640,box[2]-box[0], box[3]-box[1]))
                self.mut_items.append(rect_item)
                self.observationPlotItem.addItem(rect_item)

        self.observationPlotItem.setYRange(0, 640*self.step)
        self.bbPlotItem.setYRange(0, 640*self.step)
       
        self.reward_widget.plot(info['cumulative_reward'][0:info['step_number']],pen=(0,255,0), penSize=1, symbol='o', symbolPen=(0,255,0), symbolSize=2.5, symbolBrush=(0,255,0), clear=True)
        self.tpr_widget.plot(info['cumulative_tpr'][0:info['step_number']],pen=(0,255,0), penSize=1, symbol='o', symbolPen=(0,255,0), symbolSize=2.5, symbolBrush=(0,255,0), clear=True)
        self.fpr_widget.plot(info['cumulative_fpr'][0:info['step_number']],pen=(0,255,0), penSize=1, symbol='o', symbolPen=(0,255,0), symbolSize=2.5, symbolBrush=(0,255,0), clear=True)
        self.power_widget.plot(info['power_history'][0:info['step_number']],pen=(0,255,0), penSize=1, symbol='o', symbolPen=(0,255,0), symbolSize=2.5, symbolBrush=(0,255,0), clear=True)

        self.step += 1
        self.show()     
        QApplication.processEvents()

    def reset_bb(self):
        self.bbPlotItem.clear()

    def reset_mut(self):
        self.observationPlotItem.clear()

    def reset(self):
        self.reset_bb()
        self.reset_mut()
        self.step = 1

    def __initialize_logo_view(self):
        logo_widget = pg.GraphicsLayoutWidget()
        logo_widget.setBackground(None)
        logo_widget.ci.setContentsMargins(0, 0, 0, 0)

        logo_image_item = pg.ImageItem()
        im = plt.imread('mastiff_logo.png', format='png')
        logo_image_item.setImage(np.rot90(im,k=3))
        logo_widget.setFixedSize(int(im.shape[1]/12),int(im.shape[0]/12))
        
        logoPlotItem = logo_widget.addPlot(row=0,col=0,lockAspect=True)
        logoPlotItem.addItem(logo_image_item)
        logoPlotItem.setMouseEnabled(x=False,y=False)
        logoPlotItem.hideAxis('left')
        logoPlotItem.hideAxis('bottom')

        return logo_widget
    
    def __initialize_spectrum_view(self):
        spectrum_widget = pg.GraphicsLayoutWidget()
        spectrum_widget.setBackground(None)
        spectrum_widget.ci.setContentsMargins(0, 0, 0, 0)     
        spectrumPlotItem = spectrum_widget.addPlot(row=0,col=0,lockAspect=True)
        spectrum_image_item = pg.ImageItem() 
        spectrumPlotItem.addItem(spectrum_image_item)          
        spectrumPlotItem.setMouseEnabled(x=False,y=False)
        spectrumPlotItem.hideAxis('left')
        spectrumPlotItem.hideAxis('bottom')   

        return spectrum_widget, spectrum_image_item, spectrumPlotItem 

    def __initialize_bounding_box_view(self):
        bounding_box_widget = pg.GraphicsLayoutWidget()
        bounding_box_widget.setBackground(None)
        bounding_box_widget.ci.setContentsMargins(0, 0, 0, 0)
        bbPlotItem = bounding_box_widget.addPlot(row=0,col=0,lockAspect=True)
        bbPlotItem.setXRange(0,640+1)
        bbPlotItem.hideAxis('left')
        bbPlotItem.hideAxis('bottom')
        #bbPlotItem.setYRange(0,640+1)

        return bounding_box_widget, bbPlotItem

    def __initialize_observation_view(self):
        observation_widget = pg.GraphicsLayoutWidget()
        observation_widget.setBackground(None)
        observation_widget.ci.setContentsMargins(0, 0, 0, 0)
        observationPlotItem = observation_widget.addPlot(row=0,col=0,lockAspect=True)
        observationPlotItem.setXRange(0,640+1)
        observationPlotItem.hideAxis('left')
        observationPlotItem.hideAxis('bottom')
        #observationPlotItem.setYRange(0,640+1)

        return observation_widget, observationPlotItem

    def __initialize_cummulative_tpr_view(self):
        cummulative_tpr_view = pg.PlotWidget(lockAspect=False)
        cummulative_tpr_view.setMouseEnabled(x=False,y=False)
         # Set up the cumulative tpr per step widget.
        tick_scale = self.__get_tick_scale(self.max_steps)
        cummulative_tpr_view.setTitle("Cumulative TPR per Step", color=self.pen_color)
        cummulative_tpr_view.setXRange(0,self.max_steps-1)
        cummulative_tpr_view.setYRange(0,20)
        cummulative_tpr_view.setLabels(bottom='Step Number',left='Cumulative TPR')
        ticks_x = [*range(0,self.max_steps+1), tick_scale]
        cummulative_tpr_view.getAxis('bottom').setTicks([[(x,str(x)) for x in ticks_x]])
        cummulative_tpr_view.getAxis('bottom').setPen(self.pen_color)
        cummulative_tpr_view.getAxis('bottom').setTextPen(self.pen_color)
        ticks_y = [*range(-self.max_steps,self.max_steps+20,2)]
        cummulative_tpr_view.getAxis('left').setTicks([[(x,str(x)) for x in ticks_y]])
        cummulative_tpr_view.getAxis('left').setPen(self.pen_color)
        cummulative_tpr_view.getAxis('left').setTextPen(self.pen_color)
        cummulative_tpr_view.showAxes(True)
        return cummulative_tpr_view

    def __initialize_cummulative_fpr_view(self):
        cummulative_fpr_view = pg.PlotWidget(lockAspect=False)
        cummulative_fpr_view.setMouseEnabled(x=False,y=False)
         # Set up the cumulative tpr per step widget.
        tick_scale = self.__get_tick_scale(self.max_steps)
        cummulative_fpr_view.setTitle("Cumulative FPR per Step", color=self.pen_color)
        cummulative_fpr_view.setXRange(0,self.max_steps-1)
        cummulative_fpr_view.setYRange(0,20)
        cummulative_fpr_view.setLabels(bottom='Step Number',left='Cumulative FPR')
        ticks_x = [*range(0,self.max_steps+1), tick_scale]
        cummulative_fpr_view.getAxis('bottom').setTicks([[(x,str(x)) for x in ticks_x]])
        cummulative_fpr_view.getAxis('bottom').setPen(self.pen_color)
        cummulative_fpr_view.getAxis('bottom').setTextPen(self.pen_color)
        ticks_y = [*range(-self.max_steps,self.max_steps+20,2)]
        cummulative_fpr_view.getAxis('left').setTicks([[(x,str(x)) for x in ticks_y]])
        cummulative_fpr_view.getAxis('left').setPen(self.pen_color)
        cummulative_fpr_view.getAxis('left').setTextPen(self.pen_color)
        cummulative_fpr_view.showAxes(True)
        return cummulative_fpr_view


    def __initialize_cummulative_reward_view(self):
        cummulative_reward_view = pg.PlotWidget(lockAspect=False)
        cummulative_reward_view.setMouseEnabled(x=False,y=False)
         # Set up the cumulative reward per step widget.
        tick_scale = self.__get_tick_scale(self.max_steps)
        cummulative_reward_view.setTitle("Cumulative Reward per Step", color=self.pen_color)
        cummulative_reward_view.setXRange(0,self.max_steps-1)
        cummulative_reward_view.setYRange(-self.max_steps,0)
        cummulative_reward_view.setLabels(bottom='Step Number',left='Cumulative Reward')
        ticks_x = [*range(0,self.max_steps+1),tick_scale]
        cummulative_reward_view.getAxis('bottom').setTicks([[(x,str(x)) for x in ticks_x]])
        cummulative_reward_view.getAxis('bottom').setPen(self.pen_color)
        cummulative_reward_view.getAxis('bottom').setTextPen(self.pen_color)
        ticks_y = [*range(-self.max_steps,self.max_steps+20,2)]
        cummulative_reward_view.getAxis('left').setTicks([[(x,str(x)) for x in ticks_y]])
        cummulative_reward_view.getAxis('left').setPen(self.pen_color)
        cummulative_reward_view.getAxis('left').setTextPen(self.pen_color)
        cummulative_reward_view.showAxes(True)
        return cummulative_reward_view
    
    def __initialize_power_view(self):
        power_view = pg.PlotWidget(lockAspect=False)
        power_view.setMouseEnabled(x=False,y=False)
        tick_scale = self.__get_tick_scale(self.max_steps)
        power_view.setTitle("Power Level per Step", color=self.pen_color)
        power_view.setXRange(0,self.max_steps-1)
        power_view.setYRange(-20,20)
        power_view.setLabels(bottom='Step Number',left='Total Power')
        ticks_x = [*range(0,self.max_steps+1)]
        power_view.getAxis('bottom').setTicks([[(x,str(x)) for x in ticks_x]])
        power_view.getAxis('bottom').setPen(self.pen_color)
        power_view.getAxis('bottom').setTextPen(self.pen_color)
        ticks_y = [*range(-20,20,4)]
        power_view.getAxis('left').setTicks([[(x,str(x)) for x in ticks_y]])
        power_view.getAxis('left').setPen(self.pen_color)
        power_view.getAxis('left').setTextPen(self.pen_color)
        power_view.showAxes(True)
        return power_view

    def __get_tick_scale(self, num_indices):
        done = 0
        tick_scale = 1
        while done == 0:                
            value = int(num_indices/(5*tick_scale))
            if value >= 5:
                tick_scale = 5*tick_scale
            else:
                done = 1
        return tick_scale
