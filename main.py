#coding = 'utf-8'
import sys
from PyQt5 import QtCore, QtGui, QtWidgets,uic
from PyQt5.QtWidgets import QMessageBox
from Ui_untitled import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog,QApplication
import time
import os 
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal
import tifffile
import math
import JsonInfo
from units.image_stitch_tool import *
from units.json_tools import *
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QGroupBox, QLineEdit
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView,QGraphicsPixmapItem
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGroupBox, QLabel,QCheckBox
from PyQt5.QtWidgets import QButtonGroup,QInputDialog,QDialogButtonBox,QGraphicsScene, QGraphicsView
from units.thread import *
from PyQt5.QtGui import QPixmap,QImage
from Ui_arrangement_dialog import Ui_Dialog
from random import randint
from Ui_mip_preview import Ui_Form
import cv2
import math
from units.classify_tool import*
from units.MIP_stitched import*


class ArrangeDialog(QtWidgets.QDialog,Ui_Dialog):
    mySignal = pyqtSignal(str)

    data="1-1"
    def __init__(self):
        super(ArrangeDialog,self).__init__()
        self.setupUi(self)
        okButton = self.buttonBox.button(QDialogButtonBox.Ok)
        okButton.clicked.connect(self.accept)
        self.TypeBox.activated.connect(self.TypeBoxhandleActivated)
        self.orderBox.activated.connect(self.orderBoxhandleActivated)
    # 处理activated信号
    def TypeBoxhandleActivated(self, index):
        select_type=self.TypeBox.itemText(index)
        if(select_type=="4.filened by filename"):
            self.orderBox.clear()
            self.orderBox.addItem("Defined-by-filename")
        else:
            self.orderBox.clear()
            self.orderBox.addItem("D&R")
            self.orderBox.addItem("D&L")
            self.orderBox.addItem("U&R")
            self.orderBox.addItem("U&L")
    def orderBoxhandleActivated(self, index):
        select_order=self.orderBox.itemText(index)
        select_type=self.TypeBox.currentText()
        index=select_type[0]
        if(select_order=="D&R"):
            pixmap = QPixmap("resource/"+index+"-1.jpg")
            self.data=index+"-1"
        elif(select_order=="D&L"):
            pixmap = QPixmap("resource/"+index+"-2.jpg")
            self.data=index+"-2"
        elif(select_order=="U&R"):
            pixmap = QPixmap("resource/"+index+"-3.jpg")
            self.data=index+"-3"
        elif(select_order=="U&L"):
            pixmap = QPixmap("resource/"+index+"-4.jpg")
            self.data=index+"-4"
        elif(select_order=="Defined-by-filename"):
            pixmap = QPixmap("resource/5-1.jpg")
            self.data=index+"-1"
        scene = QGraphicsScene(self.arrangementgraphicsView)
        self.arrangementgraphicsView.setScene(scene)
        scene.addPixmap(pixmap)
        self.arrangementgraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        self.arrangementgraphicsView.show()
            
    def accept(self):  
        self.mySignal.emit(self.data)
        super().accept()


class MIPpreviewWindows(QtWidgets.QDialog,Ui_Form):
    mySignal = pyqtSignal(list)
    data="1-1"
    
    def __init__(self,jsonfile):
        super(MIPpreviewWindows,self).__init__()
        self.setupUi(self)
        self.jsonInfo=jsonfile
        self.refreshpushButton.clicked.connect(self.refresh)
        self.previewButton.clicked.connect(self.crete_mippreviw)
        # self.setMouseTracking(True)
        self.classifypushButton.clicked.connect(self.classify)
        self.submitpushButton.clicked.connect(self.submit)
        self.erode_pushButton.clicked.connect(self.erode)
        self.connect_pushButton.clicked.connect(self.connectones)
        self.classifymap = np.zeros((self.jsonInfo['row'],self.jsonInfo['col']))

        
        
        
        # self.stitch_images(self.jsonInfo['MIP_dir']+'/Z_MIP',self.jsonInfo['overlapX'],self.jsonInfo['overlapY'],self.jsonInfo["locations"])
        

    def erode(self):
        self.classifymap=erode_matrix(self.classifymap,3)
        self.make_color()
        
    def connectones(self):
        self.classifymap=connect_ones(self.classifymap)
        self.make_color()
        
    def make_color(self):
        self.rgb_image=tiff.imread(self.jsonInfo['output_path']+'/'+'MIP_preview.tiff')
        h=int(self.jsonInfo['y_length']*float(self.jsonInfo['vixel_size']/self.jsonInfo['highest_vixel_size']))
        w=int(self.jsonInfo['x_length']*float(self.jsonInfo['vixel_size']/self.jsonInfo['highest_vixel_size']))
        overlap_x=int(self.jsonInfo['overlapX']*w)
        overlap_y=int(self.jsonInfo['overlapY']*h)
        red_mask=np.zeros((h-overlap_y,w-overlap_x),dtype=np.uint8)
        red_mask[:,:]=50
        for i in range(self.jsonInfo['row']):
            for j in range(self.jsonInfo['col']):
                print(i,j)
                paste_x = (w-overlap_x)*j
                paste_y= (h-overlap_y)*i
                if(self.classifymap[i][j]==1):
                    self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,1]=self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,1]+red_mask
                if(self.classifymap[i][j]==-1):
                    self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,0]=self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,0]+red_mask
        self.show_img()
        
    def classify(self):
        indices_positve = np.argwhere(self.classifymap == 1)
        indices_negative = np.argwhere(self.classifymap == -1)
        if(len(indices_positve)<10 or len(indices_negative)<10):
            QMessageBox.warning(None, "警告", "所选样本过少（<10）") 
            return
        result=classify_compute(self.jsoninfo['output_path']+'/low_res_MIP/Z_MIP',self.jsonInfo['locations'],indices_positve,indices_negative,int(self.jsonInfo['row']),int(self.jsonInfo['col']))
        self.classifymap=result
        # result = erode_matrix(result, window_size=3)
        # result = connect_ones(result)
        self.make_color()
        QMessageBox.information(None, "提示", "已完成" )
        
    def crete_mippreviw(self):
        self.stitch_images(self.jsoninfo['output_path']+'/low_res_MIP/Z_MIP',self.jsonInfo['overlapX'],self.jsonInfo['overlapY'],self.jsonInfo["locations"])
        

    def refresh(self):
        self.rgb_image=tiff.imread(self.jsonInfo['output_path']+'/'+'MIP_preview.tiff')
        self.show_img()
        self.classifymap.fill(0)
        
    def show_img(self):
        qimage = QImage(self.rgb_image.data, self.rgb_image.shape[1], self.rgb_image.shape[0],  self.rgb_image.strides[0],QImage.Format_RGB888)
        imgShow =QPixmap.fromImage(qimage)
        self.MIPlabel.setPixmap(imgShow)
        self.MIPlabel.setScaledContents(True)

    def submit(self):
        self.classifymap.astype(int)
        if(np.all(self.classifymap == 0)):
            self.classifymap.fill(1)
        self.classifymap=self.classifymap.tolist()
        self.mySignal.emit(self.classifymap)
        self.close()

        
    def progress_bar(self,value):
        if(value<0):
            QMessageBox.warning(None, "警告", "no image") 
            return
        self.progressBar.setValue(value)
        if(value==100):
            QMessageBox.information(None, "提示", "已完成" )
        
    def stitch_images(self,folder_path, overlap_x, overlap_y,filelist):
        print(folder_path)
        self.progress_bar(0)
        raw = len(filelist)
        col = len(filelist[0])
        first_image = tiff.imread(folder_path+'/'+filelist[0][0]+'_mip_z.tiff')
        height,width = first_image.shape
        h=height
        w=width
        overlap_x=int(overlap_x*width)
        overlap_y=int(overlap_y*height)
        width=(col-1)*(width-overlap_x)+width
        height=(raw-1)*(height-overlap_y)+height
        # 初始化大图
        stitched_image = np.zeros((height,width),dtype=np.uint16)
        if(os.path.exists(self.jsonInfo['output_path']+'/'+'MIP_preview.tiff')):
            self.rgb_image=tiff.imread(self.jsonInfo['output_path']+'/'+'MIP_preview.tiff')
        else: 
            total=raw*col
            cur=0
            for y in range (raw):
                for x in range(col):
                    cur=cur+1
                    file_path=folder_path+'/'+filelist[y][x]+'_mip_z.tiff'
                    print(file_path)
                    paste_x = (w-overlap_x)*x
                    paste_y= (h-overlap_y)*y
                    image_t=tiff.imread(file_path)
                    stitched_image[paste_y:paste_y+h,paste_x:paste_x+w]=image_t
                    progress=int(cur/total*90)
                    # print(progress)
                    self.progress_bar(progress)
                    
            stitched_image=image_pre_pocess(stitched_image)

            # 合并三个单通道图像，得到三通道图像
            self.rgb_image = np.stack((stitched_image,)*3, axis=-1)
            tiff.imsave(self.jsonInfo['output_path']+'/MIP_preview.tiff',self.rgb_image)
        self.progress_bar(100)
        self.show_img()
        self.MIPlabel.mousePressEvent = self.on_label_mouse_press
        
        
    def on_label_mouse_press(self,event):
        # 获取鼠标在场景中的位置
        x, y = event.pos().x(), event.pos().y()
        print(x,y)
        w=self.MIPlabel.width()
        h=self.MIPlabel.height()
        # print(w,h)
        row=self.jsoninfo['row']
        col=self.jsoninfo['col']
        one_h=round(h/row)
        one_w=round(w/col)
        print(one_w,one_h)
        tile_x=int(x/one_w)
        tile_y=int(y/one_h)
        filelist=self.jsonInfo["locations"]
        self.rgb_image
        folder_path=self.jsoninfo['output_path']+'/low_res_MIP/Z_MIP'
        overlap_x=self.jsonInfo['overlapX']
        overlap_y=self.jsonInfo['overlapY']
        first_image = tiff.imread(folder_path+'/'+filelist[0][0]+'_mip_z.tiff')
        h,w = first_image.shape
        overlap_y=int(overlap_y*h)
        overlap_x=int(overlap_x*w)
        if(tile_x>=col or tile_y>=row):
            QMessageBox.warning(None, "警告", "超出数据块范围") 
            return
        paste_x = (w-overlap_x)*tile_x
        paste_y= (h-overlap_y)*tile_y
        red_mask=np.zeros((h-overlap_y,w-overlap_x),dtype=np.uint8)
        red_mask[:,:]=50
        if event.buttons () == QtCore.Qt.LeftButton:
            if(self.classifymap[tile_y][tile_x]==-1):
                return
            if(self.classifymap[tile_y][tile_x]==0):
                self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,1]=self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,1]+red_mask
                self.show_img()
                self.poslable.setText('所选块：({:.0f}, {:.0f})'.format(tile_x, tile_y))
                self.classifymap[tile_y][tile_x]=1
            else:
                self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,1]=self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,1]-red_mask
                self.show_img()
                self.poslable.setText('撤销块：({:.0f}, {:.0f})'.format(tile_x, tile_y))
                self.classifymap[tile_y][tile_x]=0
        if event.buttons () == QtCore.Qt.RightButton:
            if(self.classifymap[tile_y][tile_x]==1):
                return
            if(self.classifymap[tile_y][tile_x]==0):
                self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,0]=self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,0]+red_mask
                self.show_img()
                self.poslable.setText('所选块：({:.0f}, {:.0f})'.format(tile_x, tile_y))
                self.classifymap[tile_y][tile_x]=-1
            else:
                self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,0]=self.rgb_image[paste_y:paste_y+h-overlap_y,paste_x:paste_x+w-overlap_x,0]-red_mask
                self.show_img()
                self.poslable.setText('撤销块：({:.0f}, {:.0f})'.format(tile_x, tile_y))
                self.classifymap[tile_y][tile_x]=0
        


class SmartStitchwindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(SmartStitchwindow,self).__init__()
        self.w = None  # No external window yet.
        self.setupUi(self)
        self.res_labe_init(self.resgroupBox)
        for checkbox in self.resgroupBox.findChildren(QCheckBox):
            checkbox.stateChanged.connect(self.check_btn_fuc)
        # self.button_group1 = QButtonGroup()
        # self.button_group2 = QButtonGroup()
        # self.button_group3 = QButtonGroup()
        # self.button_group4 = QButtonGroup()
        # self.button_group5 = QButtonGroup()
        # self.button_group6 = QButtonGroup()
        self.InputJsonButton.clicked.connect(self.read_json)
        self.CeateJsonButton.clicked.connect(self.create_json)
        self.input_folderButton.clicked.connect(self.set_input_folder)
        self.ArrangementButton.clicked.connect(self.Getarrangement)
        self.MIPpreviewButton.clicked.connect(self.GetMIPpreview)
        self.output_folderButton.clicked.connect(self.set_output_folder)
        self.print_jsonButton.clicked.connect(self.print_dict_values)
        self.MipButton.clicked.connect(self.start_mip_task)
        self.MIP_high_res_pushButton.clicked.connect(self.start_high_res_mip_task)
        self.xy_shift_pushButton.clicked.connect(self.start_xy_shift_task)
        self.high_res_x_y_shift_pushButton.clicked.connect(self.start_high_res_xy_shift_task)
        self.low_res_z_shift_pushButton.clicked.connect(self.start_z_shift_task)
        self.high_res_z_shift_pushButton.clicked.connect(self.start_high_res_z_shift_task)
        self.high_res_op_pushButton.clicked.connect(self.start_high_res_op_task)
        self.low_res_op_pushButton.clicked.connect(self.start_low_res_op_task)
        self.HR_shift_floatToint_pushButton.clicked.connect(self.HR_shift_floatToint)
        self.LR_shift_floatToint_pushButton.clicked.connect(self.LR_shift_floatToint)
        self.LR_stitched_MIP_pushButton.clicked.connect(self.LR_stitched_MIP)
        self.HR_stitched_MIP_pushButton.clicked.connect(self.HR_stitched_MIP)
        self.make_slice_LR_pushButton.clicked.connect(self.make_slice_LR)
        self.make_slice_HR_pushButton.clicked.connect(self.make_slice_HR)
        self.LR_z_slice_pushButton.clicked.connect(self.output_LR_zslice)
        self.HR_z_slice_pushButton.clicked.connect(self.output_HR_zslice)
        self.output_terafly_pushButton.clicked.connect(self.output_terafly)
        self.test_threadpushButton.clicked.connect(self.test)


        
        
        self.high_res_input_folder_Button.clicked.connect(self.set_highest_res_folder)
        self.actionopen.triggered.connect(self.read_json)
        # self.CeateJsonButton.setAutoRepeatclicked.connect(True) # 设置为允许自动重复触发
        self.jsoninfo={
                "json_path":'',
                "input_path": '',
                "output_path": '',
                "highest_path": '',
                "wildname": '',
                "x_length":0,
                "y_length":0,
                "z_length":0,
                "vixel_size":1.,
                "highest_vixel_size":1.,
                "overlapX":0.2,
                "overlapY":0.2,
                "overlapZ":0.2,
                "row":0,
                "col":0
        }


    def Getarrangement(self):
        child = ArrangeDialog()
        child.mySignal.connect(self.handleArrangeDataChanged)       
        child.exec()
        child.close()   

    def GetMIPpreview(self):
        child = MIPpreviewWindows(self.jsoninfo)
        child.jsoninfo=self.jsoninfo
        child.mySignal.connect(self.handleMIPshow)       
        child.exec()
        child.close() 
        # mip_image=stitch_images(self.jsoninfo['MIP_dir']+'/Z_MIP',self.jsoninfo['overlapX'],self.jsoninfo['overlapY'],self.jsoninfo["locations"])  
        # tiff.imsave("mipprevie.tiff",mip_image)

    def handleArrangeDataChanged(self, data):
            # Do something with the data received from the subwindow
            print("Received data from subwindow:", data)
            locations=stitch_list(self.jsoninfo['input_path'],data,self.jsoninfo['col'],self.jsoninfo['row'])
            self.jsoninfo["ArrangeMethod"]=data
            self.jsoninfo["locations"]=locations

    def handleMIPshow(self, data):
            # Do something with the data received from the subwindow
            print("Received data from subwindow:", data)
            self.jsoninfo["classifymap"]=data
            self.jsoninfo["high_res_location"]=[]
            for i in range (self.jsoninfo['row']):
                line=[]
                for j in range(self.jsoninfo['col']):
                    file_name=self.jsoninfo["locations"][i][j]
                    file_exit='.tiff'
                    if(os.path.exists(self.jsoninfo['highest_path']+'/'+file_name+file_exit) and self.jsoninfo["classifymap"][i][j]==1 ):
                        line.append(file_name)
                    else:
                        line.append('None')
                self.jsoninfo["high_res_location"].append(line)

    # 读取json数据到主窗口
    def read_json(self):
        # 创建一个文件选择对话框
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("JSON文件 (*.json)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        # 设置关闭按钮行为
        file_dialog.setWindowFlags(file_dialog.windowFlags() | Qt.WindowStaysOnTopHint | Qt.WindowCloseButtonHint)
        # 显示对话框并获取所选文件路径
        if file_dialog.exec_() == QFileDialog.Accepted:
            ret=self.loadjson(file_dialog.selectedFiles()[0])
            if(ret>0):
                self.refresh_groupbox(self.JsonInfoBox)
                QMessageBox.information(self, '提示', '读取json成功！')
            else:
                QMessageBox.information(self, '提示', '打开失败！')
                

                
    def check_btn_fuc(self):
        if  (not 'res_list' in self.jsoninfo):
            QMessageBox.information(self, '提示', '没有加载json')
            return 
        for checkbox in self.resgroupBox.findChildren(QCheckBox):
            if checkbox.objectName() == "all_stitch1":
                if checkbox.checkState() == Qt.Checked:
                    self.part_stitch1.setEnabled(False)
                    print(self.jsoninfo["res_list"]["0"]["all_stitch"])
                    self.jsoninfo["res_list"]["0"]["all_stitch"]=True
                    print(self.jsoninfo["res_list"]["0"]["all_stitch"])
                else:
                    self.part_stitch1.setEnabled(True)
                    self.jsoninfo["res_list"]["0"]["all_stitch"]=False
            if checkbox.objectName() == "all_stitch2":
                if checkbox.checkState() == Qt.Checked:
                    self.part_stitch2.setEnabled(False)
                    self.jsoninfo["res_list"]["1"]["all_stitch"]=True
                else:
                    self.part_stitch2.setEnabled(True)
                    self.jsoninfo["res_list"]["1"]["all_stitch"]=False
            if checkbox.objectName() == "all_stitch3":
                if checkbox.checkState() == Qt.Checked:
                    self.part_stitch3.setEnabled(False)
                    self.jsoninfo["res_list"]["2"]["all_stitch"]=True
                else:
                    self.part_stitch3.setEnabled(True)
                    self.jsoninfo["res_list"]["2"]["all_stitch"]=False
            if checkbox.objectName() == "all_stitch4":
                if checkbox.checkState() == Qt.Checked:
                    self.part_stitch4.setEnabled(False)
                    self.jsoninfo["res_list"]["3"]["all_stitch"]=True
                else:
                    self.part_stitch4.setEnabled(True)
                    self.jsoninfo["res_list"]["3"]["all_stitch"]=False
            if checkbox.objectName() == "all_stitch5":
                if checkbox.checkState() == Qt.Checked:
                    self.part_stitch5.setEnabled(False)
                    self.jsoninfo["res_list"]["4"]["all_stitch"]=True
                else:
                    self.part_stitch5.setEnabled(True)
                    self.jsoninfo["res_list"]["4"]["all_stitch"]=False
            if checkbox.objectName() == "all_stitch6":
                if checkbox.checkState() == Qt.Checked:
                    self.part_stitch6.setEnabled(False)
                    self.jsoninfo["res_list"]["5"]["all_stitch"]=True
                else:
                    self.part_stitch6.setEnabled(True)
                    self.jsoninfo["res_list"]["5"]["all_stitch"]=False

            if checkbox.objectName() == "part_stitch1":
                if checkbox.checkState() == Qt.Checked:
                    self.all_stitch1.setEnabled(False)
                    self.jsoninfo["res_list"]["0"]["part_atitch"]=True
                else:
                    self.all_stitch1.setEnabled(True)
                    self.jsoninfo["res_list"]["0"]["part_atitch"]=False
            if checkbox.objectName() == "part_stitch2":
                if checkbox.checkState() == Qt.Checked:
                    self.all_stitch2.setEnabled(False)
                    self.jsoninfo["res_list"]["1"]["part_atitch"]=True
                else:
                    self.all_stitch2.setEnabled(True)
                    self.jsoninfo["res_list"]["1"]["part_atitch"]=False
            if checkbox.objectName() == "part_stitch3":
                if checkbox.checkState() == Qt.Checked:
                    self.all_stitch3.setEnabled(False)
                    self.jsoninfo["res_list"]["2"]["part_atitch"]=True
                else:
                    self.all_stitch3.setEnabled(True)
                    self.jsoninfo["res_list"]["2"]["part_atitch"]=False
            if checkbox.objectName() == "part_stitch4":
                if checkbox.checkState() == Qt.Checked:
                    self.all_stitch4.setEnabled(False)
                    self.jsoninfo["res_list"]["3"]["part_atitch"]=True
                else:
                    self.all_stitch4.setEnabled(True)
                    self.jsoninfo["res_list"]["3"]["part_atitch"]=False
            if checkbox.objectName() == "part_stitch5":
                if checkbox.checkState() == Qt.Checked:
                    self.all_stitch5.setEnabled(False)
                    self.jsoninfo["res_list"]["4"]["part_atitch"]=True
                else:
                    self.all_stitch5.setEnabled(True)
                    self.jsoninfo["res_list"]["4"]["part_atitch"]=False
            if checkbox.objectName() == "part_stitch6":
                if checkbox.checkState() == Qt.Checked:
                    self.all_stitch6.setEnabled(False)
                    self.jsoninfo["res_list"]["5"]["part_atitch"]=True
                else:
                    self.all_stitch6.setEnabled(True)
                    self.jsoninfo["res_list"]["5"]["part_atitch"]=False
    
    

    def loadjson(self,selected_file):
            # 读取所选文件
        with open(selected_file, 'r') as f:
            data = json.load(f)
        # 判断新字典是否是原字典的子集
        # 更新原字典
        self.jsoninfo.update(data)
        self.jsoninfo["json_path"]=selected_file
        if(self.jsoninfo["output_path"]==""):
            self.jsoninfo["output_path"]==os.path.dirname(selected_file)
        if(self.jsoninfo["input_path"]!="" and os.path.exists(self.jsoninfo["input_path"]) and os.path.isdir(self.jsoninfo["input_path"]) and self.jsoninfo["wildname"]==""):
            wild_name=find_same_name(self.jsoninfo["input_path"])
            self.jsoninfo["wildname"]=wild_name
        self.set_Res_label()
        return 1 

        
        
        
    def set_input_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, 'Select Folder', '/')
        self.input_path.setText(folder_path)
        
    def set_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, 'Select Folder', '/')
        self.output_path.setText(folder_path)

    def set_highest_res_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, 'Select Folder', '/')
        self.high_res_input_path.setText(folder_path)
        
        

        

    # 根据编辑框制造一个json文件
    def create_json(self):
        self.syn_groupbox(self.JsonInfoBox)
        if(self.jsoninfo["output_path"]==""):
            QMessageBox.warning(self, '警告', '未指定输出路径！')
        else:
            if(os.path.exists(self.jsoninfo["json_path"])):
                reply = QMessageBox.question(None, 'Message', '确定要覆写文件中的json信息吗', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    print(self.jsoninfo["json_path"])
                    ret=generate_json_file(self.jsoninfo["json_path"],self.jsoninfo)
                    self.loadjson(self.jsoninfo["json_path"])
                    if(ret>0):
                        QMessageBox.information(self, '提示', '已更新json文件！')
                    else:
                        QMessageBox.information(self, '提示', '覆写失败')
            else:
                if os.path.exists(self.jsoninfo["output_path"]) and os.path.isdir(self.jsoninfo["output_path"]):
                    self.jsoninfo["json_path"]=self.jsoninfo["output_path"]+'/datainfo.json'          
                    ret=generate_json_file(self.jsoninfo["json_path"],self.jsoninfo)
                    if(ret>0):
                        self.loadjson(self.jsoninfo["json_path"])
                        self.refresh_groupbox(self.JsonInfoBox)
                        QMessageBox.information(self, '提示', '已保存json文件！')
                        
                    else:
                        QMessageBox.information(self, '提示', '保存失败')
                else:
                    QMessageBox.warning(self, '警告', '路径不合法！')
                    

                
                
                
                
            
                
            

    #刷新编辑框各个信息的值   
    def refresh_groupbox(self,groupbox: QGroupBox):
        for widget in groupbox.findChildren(QLineEdit):
            widget.clear()
            if widget.objectName() == "input_path":
                widget.setText(self.jsoninfo["input_path"])
            elif widget.objectName() == "output_path":
                widget.setText(self.jsoninfo["output_path"])
            elif widget.objectName() == "highest_path":
                widget.setText(self.jsoninfo["highest_path"])
            elif widget.objectName() == "x_lengthEdit":
                widget.setText(str(self.jsoninfo["x_length"]))
            elif widget.objectName() == "y_lengthEdit":
                widget.setText(str(self.jsoninfo["y_length"]))
            elif widget.objectName() == "z_lengthEdit":
                widget.setText(str(self.jsoninfo["z_length"]))
            elif widget.objectName() == "overlapXEdit":
                widget.setText(str(self.jsoninfo["overlapX"]))
            elif widget.objectName() == "overlapZEdit":
                widget.setText(str(self.jsoninfo["overlapZ"]))
            elif widget.objectName() == "overlapYEdit":
                widget.setText(str(self.jsoninfo["overlapY"]))
            elif widget.objectName() == "wildnameEdit":
                widget.setText(str(self.jsoninfo["wildname"]))
            elif widget.objectName() == "vixel_sizeEdit":
                widget.setText(str(self.jsoninfo["vixel_size"]) )
            elif widget.objectName() == "highest_vixel_sizeEdit":
                widget.setText(str(self.jsoninfo["highest_vixel_size"]) )
            elif widget.objectName() == "rowlineEdit":
                widget.setText(str(self.jsoninfo["row"]) )
            elif widget.objectName() == "collineEdit":
                widget.setText(str(self.jsoninfo["col"]) )
        # self.rowtextEdit.setText(str(self.jsoninfo["row"]) )
        # self.coltextEdit.setText(str(self.jsoninfo["col"]) )
        
                
    # 将编辑框里面的值同步到jsonfile
    def syn_groupbox(self,groupbox: QGroupBox):
        for widget in groupbox.findChildren(QLineEdit):
            if widget.objectName() == "input_path" :
                if(widget.text()!=""):
                    self.jsoninfo["input_path"]=widget.text()
            elif widget.objectName() == "output_path":
                if(widget.text()!=""):
                    self.jsoninfo["output_path"]=widget.text()
            elif widget.objectName() == "high_res_input_path":
                if(widget.text()!=""):
                    self.jsoninfo["highest_path"]=widget.text()
            elif widget.objectName() == "x_lengthEdit":
                if(widget.text()!=""):
                    self.jsoninfo["x_length"]=int(widget.text())
            elif widget.objectName() == "y_lengthEdit":
                if(widget.text()!=""):
                    self.jsoninfo["y_length"]=int(widget.text())
            elif widget.objectName() == "z_lengthEdit":
                if(widget.text()!=""):
                    self.jsoninfo["z_length"]=int(widget.text())
            elif widget.objectName() == "overlapXEdit":
                if(widget.text()!=""):
                    self.jsoninfo["overlapX"]=float(widget.text())
            elif widget.objectName() == "overlapZEdit":
                if(widget.text()!=""):
                    self.jsoninfo["overlapZ"]=float(widget.text())
            elif widget.objectName() == "overlapYEdit":
                if(widget.text()!=""):
                    self.jsoninfo["overlapY"]=float(widget.text())
            elif widget.objectName() == "wildnameEdit":
                if(widget.text()!=""):
                    self.jsoninfo["wildname"]=widget.text()
            elif widget.objectName() == "vixel_sizeEdit":
                if(widget.text()!=""):
                    self.jsoninfo["vixel_size"]=float(widget.text())
            elif widget.objectName() == "highest_vixel_sizeEdit":
                if(widget.text()!=""):
                    self.jsoninfo["highest_vixel_size"]=float(widget.text())
            elif widget.objectName() == "rowlineEdit":
                if(widget.text()!=""):
                    self.jsoninfo["row"]=int(widget.text())
            elif widget.objectName() == "collineEdit":
                if(widget.text()!=""):
                    self.jsoninfo["col"]=int(widget.text())
        
        
           

    def choose_directory():
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_() == QFileDialog.Accepted:
            directory = dialog.selectedFiles()[0]
            QMessageBox.information(None, "提示", "您选择的文件夹路径为：" + directory)
            return directory
        else:
            QMessageBox.warning(None, "警告", "您取消了选择文件夹路径") 
            return ""   
        
    def setRes(self):
        if('res_list' in self.jsoninfo):
            return
        multiple=int(self.jsoninfo['highest_vixel_size']/self.jsoninfo['vixel_size'])
        if(multiple>32):
            self.jsoninfo['vixel_size']=self.jsoninfo['highest_vixel_size']
            QMessageBox.warning(None, "警告", "分辨率相差过大!")
            return -1
        img=tiff.imread(self.jsoninfo['highest_path']+'/'+os.listdir(self.jsoninfo['highest_path'])[0])
        max_x=img.shape[2]
        max_y=img.shape[1]
        max_z=img.shape[0]
        img=tiff.imread(self.jsoninfo['input_path']+'/'+os.listdir(self.jsoninfo['input_path'])[0])
        lr_x=img.shape[2]
        lr_y=img.shape[1]
        lr_z=img.shape[0]
        res_table={}
        for cur_res in range(0,6):
            if(pow(2,cur_res)<multiple):
                now_x=int(max_x/pow(2,cur_res))
                now_y=int(max_y/pow(2,cur_res))
                now_z=int(max_z/pow(2,cur_res))
            else:
                now_x=int(lr_x/pow(2,cur_res)*multiple)
                now_y=int(lr_y/pow(2,cur_res)*multiple)
                now_z=int(lr_z/pow(2,cur_res)*multiple)    
            if(now_x<=0 or now_y<=0 or now_x<=0):
                break
            str_res=str(now_x)+'x'+str(now_y)+'x'+str(now_z)
            print(str_res)
            onedate={str(cur_res): {'res': str_res, 'all_stitch': False, 'part_atitch': False}}
            res_table.update(onedate)
        self.jsoninfo["res_list"]=res_table
        print(self.jsoninfo)

    def set_Res_label(self):
        self.setRes()
        labels = self.resgroupBox.findChildren(QLabel)
        for label in labels:
            if label.objectName()=="res1_label":
                label.setText(self.jsoninfo["res_list"]["0"]['res'])
            if label.objectName()=="res2_label":
                label.setText(self.jsoninfo["res_list"]["1"]['res'])
            if label.objectName()=="res3_label":
                label.setText(self.jsoninfo["res_list"]["2"]['res'])
            if label.objectName()=="res4_label":
                label.setText(self.jsoninfo["res_list"]["3"]['res'])
            if label.objectName()=="res5_label":
                label.setText(self.jsoninfo["res_list"]["4"]['res'])
            if label.objectName()=="res6_label":
                label.setText(self.jsoninfo["res_list"]["5"]['res'])               
                
    def res_labe_init(self,groupbox: QGroupBox):
        labels = groupbox.findChildren(QLabel)
        for label in labels:
            label.setText('** x ** x **')

    def print_dict_values(self):
        json_str = json.dumps(self.jsoninfo, indent=4)
        self.jsonBrowser.clear()
        self.jsonBrowser.setText(json_str)
        
    # def blind_select_gruop_init(self):

    #     self.button_group1.addButton(self.all_stitch1,1)
    #     self.button_group1.addButton(self.part_stitch1,2)
    #     self.button_group1.setExclusive(True)

    #     self.button_group2.addButton(self.all_stitch2,1)
    #     self.button_group2.addButton(self.part_stitch2,2)
    #     self.button_group2.setExclusive(True)

    #     self.button_group3.addButton(self.all_stitch3,1)
    #     self.button_group3.addButton(self.part_stitch3,2)
    #     self.button_group3.setExclusive(True)

    #     self.button_group4.addButton(self.all_stitch4,1)
    #     self.button_group4.addButton(self.part_stitch4,2)
    #     self.button_group4.setExclusive(True)

    #     self.button_group5.addButton(self.all_stitch5,1)
    #     self.button_group5.addButton(self.part_stitch5,2)
    #     self.button_group5.setExclusive(True)

    #     self.button_group6.addButton(self.all_stitch6,1)
    #     self.button_group6.addButton(self.part_stitch6,2)
    #     self.button_group6.setExclusive(True)

    def make_slice_LR(self):
        print('Start clicked.')
        mip_path=self.jsoninfo['output_path']+'/low_res_MIP/X_MIP'
        namelisft_X_mip=os.listdir(mip_path )
        sp=tiff.imread(mip_path+'/'+namelisft_X_mip[0]).shape
        z_length =sp[0] 
        self.thread = make_slice(locations=self.jsoninfo['locations'],input_folder=self.jsoninfo['input_path'],z_length=z_length,save_dirpath=self.jsoninfo['output_path']+'/LR_slices')  #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def make_slice_HR(self):
        print('Start clicked.')
        mip_path=self.jsoninfo['output_path']+'/high_res_MIP/X_MIP'
        namelisft_X_mip=os.listdir(mip_path )
        sp=tiff.imread(mip_path+'/'+namelisft_X_mip[0]).shape
        z_length =sp[0] 
        self.thread = make_slice(locations=self.jsoninfo['high_res_location'],input_folder=self.jsoninfo['highest_path'],z_length=z_length,save_dirpath=self.jsoninfo['output_path']+'/HR_slices')  #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progressBar2)
        # 启动线程，执行线程类中run函数
        self.thread.start()

    def start_mip_task(self):
        print('Start clicked.')
        self.thread = MIP_Thread(data=self.jsoninfo,input_path=self.jsoninfo['input_path'],second_path='low_res_MIP',locations=self.jsoninfo['locations'])  #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        # 启动线程，执行线程类中run函数
        self.thread.start()

    def start_high_res_mip_task(self):
        print('Start clicked.')
        self.thread = MIP_Thread(data=self.jsoninfo,input_path=self.jsoninfo['highest_path'],second_path='high_res_MIP',locations=self.jsoninfo['high_res_location'])  #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        # 启动线程，执行线程类中run函数
        self.thread.start()

    def start_xy_shift_task(self):
        print('Start clicked.')
        self.thread = shiftcompute_Thread(data=self.jsoninfo,input_folder=self.jsoninfo['output_path']+'/low_res_MIP',location=self.jsoninfo['locations'],outputname='low_res_x_y_shift.json')  #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        self.thread.start()
        # self.thread = Z_shift_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # # 启动线程，执行线程类中run函数
        # self.thread.start()
        
    def start_high_res_xy_shift_task(self):
        print('Start clicked.')
        self.thread = shiftcompute_Thread(data=self.jsoninfo,input_folder=self.jsoninfo['output_path']+'/high_res_MIP',location=self.jsoninfo['high_res_location'],outputname='high_res_x_y_shift.json')  #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        self.thread.start()
        # self.thread = Z_shift_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # # 启动线程，执行线程类中run函数
        # self.thread.start()

    def start_z_shift_task(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = Z_shift_Thread(data=self.jsoninfo,input_folder=self.jsoninfo['output_path']+'/low_res_MIP',location=self.jsoninfo['locations'],outputname='low_res_z_shift.json')  #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progressBar2)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def start_high_res_z_shift_task(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = Z_shift_Thread(data=self.jsoninfo,input_folder=self.jsoninfo['output_path']+'/high_res_MIP',location=self.jsoninfo['high_res_location'],outputname='high_res_z_shift.json')  #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progressBar2)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def start_high_res_op_task(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = gloabal_opi_Thread(data=self.jsoninfo,isHighres=true) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def start_low_res_op_task(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = gloabal_opi_Thread(data=self.jsoninfo,isHighres=false) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def HR_shift_floatToint(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = float2int_Thread(data=self.jsoninfo,isHigh=true) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def LR_shift_floatToint(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = float2int_Thread(data=self.jsoninfo,isHigh=false) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progressBar2)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def LR_stitched_MIP(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = make_LR_MIP_stitched(data=self.jsoninfo) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def HR_stitched_MIP(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = make_HR_MIP_stitched(data=self.jsoninfo) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progressBar2)
        # 启动线程，执行线程类中run函数
        self.thread.start()

    def output_LR_zslice(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = LR_make_slice(data=self.jsoninfo) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progress_bar)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def output_HR_zslice(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = HR_make_slice(data=self.jsoninfo) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progressBar2)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def output_terafly(self):
        print('Start clicked.')
        # self.thread = shiftcompute_Thread(data=self.jsoninfo)  #实例化一个线程，参数t设置为jsoninfos
        # # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        # self.thread.finishSignal.connect(self.update_progress_bar)
        # self.thread.start()
        self.thread = terafly_output_thread(data=self.jsoninfo) #实例化一个线程，参数t设置为jsoninfos
        # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
        self.thread.finishSignal.connect(self.update_progressBar2)
        # 启动线程，执行线程类中run函数
        self.thread.start()
        
    def test(self):
        multiple=int(self.jsoninfo['highest_vixel_size']/self.jsoninfo['vixel_size'])
        cur_res=2
        x = self.jsoninfo['res_list'][str(cur_res)]['res'].split("x")
        out_terafly_part=output_terafly_c(data=self.jsoninfo,posjson_path=self.jsoninfo['output_path']+'/HR-z_y_x_p.json',input_folder=self.jsoninfo['highest_path'],
                                            result_folder=self.jsoninfo['output_path']+'/result_terafly',x_length=int(x[0]),y_length=int(x[1]),planes_num_0=int(x[2]),rescale_factor=int(pow(2,cur_res)),real_factors=pow(2,cur_res))
        out_terafly_part.run()


        


    def update_progress_bar(self,value):
        if(value<0):
            QMessageBox.warning(None, "警告", "no image") 
            return
        self.update_bar.setValue(value)
        if(value==100):
            QMessageBox.information(None, "提示", "已完成" )
            if(os.path.exists( self.jsoninfo['output_path']+'/'+"low_res_x_y_shift.json") and not 'low_res_xy_shift' in self.jsoninfo):
                json_env = self.jsoninfo['output_path']+'/'+"low_res_x_y_shift.json"
                with open(json_env, 'r')as fp:
                    data=json.load(fp)
                    self.jsoninfo['low_res_xy_shift'] =  data['result_y_x_s']
            if(os.path.exists( self.jsoninfo['output_path']+'/'+"high_res_x_y_shift.json") and not 'high_res_xy_shift' in self.jsoninfo):
                json_env = self.jsoninfo['output_path']+'/'+"high_res_x_y_shift.json"
                with open(json_env, 'r')as fp:
                    data=json.load(fp)
                    self.jsoninfo['high_res_xy_shift'] =  data['result_y_x_s']
            
    def update_progressBar2(self,value):
        if(value<0):
            QMessageBox.warning(None, "警告", "no image") 
            return
        self.progressBar2.setValue(value)
        if(value==100):
            QMessageBox.information(None, "提示", "已完成" )
            if(os.path.exists( self.jsoninfo['output_path']+'/'+"low_res_z_shift.json") and not 'low_res_z_shift' in self.jsoninfo):
                json_env = self.jsoninfo['output_path']+'/'+"low_res_z_shift.json"
                with open(json_env, 'r')as fp:
                    data=json.load(fp)
                    self.jsoninfo['low_res_z_shift'] = data['result_z_yx_s']
            if(os.path.exists( self.jsoninfo['output_path']+'/'+"high_res_z_shift.json") and not 'high_res_z_shift' in self.jsoninfo):
                json_env = self.jsoninfo['output_path']+'/'+"high_res_z_shift.json"
                with open(json_env, 'r')as fp:
                    data=json.load(fp)
                    self.jsoninfo['high_res_z_shift'] = data['result_z_yx_s']
                
                
        
            
            
        
    


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = SmartStitchwindow()
    window.show()
    sys.exit(app.exec_())