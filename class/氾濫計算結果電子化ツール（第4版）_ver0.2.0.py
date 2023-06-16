"""
「氾濫計算結果電子化ツール（第4版）」
base
最終更新日: 2023/05/xx

"""

import os
import sys
import pandas as pd
import numpy as np
import datetime
import threading
import pickle
from pprint import pprint

from glob import glob
from tqdm import tqdm

import PySimpleGUI as sg
from queue import Queue
#import yaml
#import codecs

from layout import input_frame
from layout import mod_gui

from modules import mod_tools
from modules import mod_ctl
from modules import mod_case_result
from modules import mod_kensaku_time
from modules import mod_mk_official
from modules import mod_mk_breakpoint

#############################################################################
def main():

    ## 作業のキューの設定
    que_ui = Queue()
    que_data = Queue()    
    
    ## GUIの作成
    layout, cal_list, maxall_list, maxalltime_list = input_frame.generate()
    
    window = sg.Window("氾濫計算結果を用いた電子化ガイドラインデータ作成ツール（第4版）", layout, finalize=True)
    
            
            
    ## イベントの実行
    while True:
        event, values = window.read()
        
        ## 終了処理
        if event in (sg.WIN_CLOSED, None):
            break
        
        ## 計算結果入力部分
        elif event == "-Add_cal-":
            if values["scale_cal"] == "" or values["cal"] == "" or values["hatei"] == "":
                sg.popup("[ERROR] 入力項目に不備があります")
            else:
                cal_list = mod_gui.save_info_list(cal_list, values["scale_cal"], values["cal"], values["hatei"])
                window["cal_table"].update(cal_list)

            window["scale_cal"].update("")
            window["cal"].update("")
            window["hatei"].update("")

                
        elif event == "-Delete_cal-":
            
            if len(values["cal_table"]) == 0:
                continue
            
            del_list = values["cal_table"]
            cal_list = [cal_list[j] for j in range(len(cal_list)) if j not in del_list]
            window["cal_table"].update(cal_list)
            
        
        ## MAXALL入力部分
        elif event == "-Add_maxall-":
            if values["scale_maxall"] == "" or values["meshsize_maxall"] == "" or values["res_maxall"] == "":
                sg.popup("[ERROR] 入力項目に不備があります")
            else:
                maxall_list = mod_gui.save_info_list(maxall_list, values["scale_maxall"], values["meshsize_maxall"], values["res_maxall"])
                window["maxall_table"].update(maxall_list)

            window["scale_maxall"].update("")
            window["meshsize_maxall"].update("")
            window["res_maxall"].update("")
                        
        elif event == "-Delete_maxall-":
            
            if len(values["maxall_table"]) == 0:
                continue
            
            del_list = values["maxall_table"]
            maxall_list = [maxall_list[j] for j in range(len(maxall_list)) if j not in del_list]
            window["maxall_table"].update(maxall_list)
            

        ## MAXALL_TIME入力部分
        elif event == "-Add_maxalltime-":
            if values["scale_maxalltime"] == "" or values["meshsize_maxalltime"] == "" or values["res_maxalltime"] == "":
                sg.popup("[ERROR] 入力項目に不備があります")
            else:
                maxalltime_list = mod_gui.save_info_list(maxalltime_list, values["scale_maxalltime"], values["meshsize_maxalltime"], values["res_maxalltime"])
                window["maxalltime_table"].update(maxalltime_list)

            window["scale_maxalltime"].update("")
            window["meshsize_maxalltime"].update("")
            window["res_maxalltime"].update("")
            
        elif event == "-Delete_maxalltime-":
            
            if len(values["maxalltime_table"]) == 0:
                continue
            
            del_list = values["maxalltime_table"]
            maxalltime_list = [maxalltime_list[j] for j in range(len(maxalltime_list)) if j not in del_list]
            window["maxalltime_table"].update(maxalltime_list)

        ## すべて実行
        elif event in "-RUN-":
            
#            print(values)
            values["maxall_list"] = maxall_list
            values["maxalltime_list"] = maxalltime_list
            values["cal_list"] = cal_list
            # ログファイルの保存
            with open(r"bin/log.bin", "wb") as p:
                pickle.dump(values, p)
                        
            mod_gui.change_dms(values["d_lon"], values["m_lon"], values["s_lon"], values, "d_lon", "m_lon", "s_lon", "lon_ori")
            mod_gui.change_dms(values["d_lat"], values["m_lat"], values["s_lat"], values, "d_lat", "m_lat", "s_lat", "lat_ori")
            values["mesh_size"] = int(values["mesh_size"])
            values["mesh_xmax"] = int(values["mesh_xmax"])
            values["mesh_ymax"] = int(values["mesh_ymax"])
            values["hs_min"] = float(values["hs_min"])
            if("標準版" in values["hs_hanrei"]):
                values["hs_hanrei"] = 0
            elif("詳細版" in values["hs_hanrei"]):
                values["hs_hanrei"] = 1
                
            if("あり" in values["minus_flag"]):
                values["minus_flag"] = 1
            elif("なし" in values["minus_flag"]):
                values["minus_flag"] = 0

            ## MAXALL_LISTの読み込み用更新
            read_maxall_list = []
            for i in range(len(values["maxall_list"])):
                temp_list = []
                for j in range(4):
                    if(j==0):
                        temp_list.append(values["maxall_list"][i][j])
                    elif(j==1):
                        temp_list.append("H")
                    elif(j>=2):
                        temp_list.append(values["maxall_list"][i][j-1])
                read_maxall_list.append(temp_list)
            values["maxall_list"] = read_maxall_list

            
            ## MAXALLTIME_LISTの読み込み用更新
            read_maxalltime_list = []
            for i in range(len(values["maxalltime_list"])):
                temp_list = []
                for j in range(4):
                    if(j==0):
                        temp_list.append(values["maxalltime_list"][i][j])
                    elif(j==1):
                        temp_list.append("T")
                    elif(j>=2):
                        temp_list.append(values["maxalltime_list"][i][j-1])
                read_maxalltime_list.append(temp_list)
            values["maxalltime_list"] = read_maxalltime_list

            
            kibo_file = [values["cal_list"][i][0] for i in range(len(values["cal_list"]))]
            kibo_calset_path = [values["cal_list"][i][1] for i in range(len(values["cal_list"]))]
            kibo_hateipt_path = [values["cal_list"][i][2] for i in range(len(values["cal_list"]))]
            temp = []
            for data in kibo_file:
                if("L" not in data):
                    temp.append(data.zfill(4))
                else:
                    temp.append(data)
            kibo_file = temp
                

            values["kibo_calset_dict"] = dict(zip(kibo_file, kibo_calset_path))
            values["kibo_hateipt_dict"] = dict(zip(kibo_file, kibo_hateipt_path))

            try:
                run(values)
                if(values["FolderIndex"] == True):
                    sg.popup("正常に終了しました。","※注意※","「FolderIndex.CSV」(全体管理用メタデータ)、「METADATA.CSV」（各ケースメタデータ）の内容はダミーで出力しております。","各項目は提出先に確認し、必要に応じて修正してください。","また、「DZONE」フォルダは公表図と整合の取れたGISデータを格納してください。")
                else:
                    sg.popup("正常に終了しました。","※注意※","「METADATA.CSV」（各ケースメタデータ）の内容はダミーで出力しております。","各項目は提出先に確認し、必要に応じて修正してください。","また、「DZONE」フォルダは公表図と整合の取れたGISデータを格納してください。")
            except:
                sg.popup("エラーが発生しました")
#                print("ERROR")


#############################################################################
def run(values):
    
#    with codecs.open(r"bin/log_values.yaml", "w", "utf-8") as y:
#        yaml.dump(values, y, encoding="utf-8", allow_unicode=True)    

    ## 出力用フォルダ作成
    kouzui_kibo, folderindex_fol = mod_ctl.make_v4_folder(values)
    
    ## ベース出力フォルダの作成
    mod_ctl.make_kouzui_official_folder(kouzui_kibo)
    
    
    ## 各外力規模でループ処理
    kibo_dict = values.get("kibo_calset_dict")
    
    ## 各ケースの結果変換「BPxxxxxフォルダ内の時系列結果ファイル作成」
    for fol, output_path in kouzui_kibo.items():
        print("[CONV]   ", str(fol))
        read_path = kibo_dict[fol]
        if(values["CASE"] == True):
            pprint("CASE_RESULT")
            mod_case_result.run(fol, values, read_path, output_path)
        if(values["KENSAKU"] == True):
            mod_kensaku_time.run(fol, values, read_path, output_path)
    
    ## MAXALL,OFFICIALフォルダの作成
    if(values["MAXALL"] == True):
        mod_mk_official.run(values, read_path, kouzui_kibo, values["SHAPE"])
        

        
    ## FolderIndex.CSVの出力
    if(values["FolderIndex"] == True):
        mod_tools.mk_folderindex(folderindex_fol)
        

#############################################################################
if __name__ == "__main__":
    
    ## GUIの立ち上げ
    main()
