from GA_bp.GA import GA
from LSTM.lstm import My_LSTM
from  My_SVR.svr import My_SVR
import numpy as np
from tkinter import Tk, Button, Frame, TOP, W, YES, X, BOTH, LEFT, PhotoImage, Label, filedialog, \
    BOTTOM, CENTER


def load_data(filename='sample/data.csv'):
    data = []
    i = 1
    with open(filename, 'r') as f:
        for line in f:
            sample = line.strip().split(',')
            if len(sample) == 15 and i >= 577:
                data.append([float(sample[1]), float(sample[2]), float(sample[3]),
                            float(sample[4]), float(sample[5]), float(sample[6]),
                            float(sample[7]), float(sample[8]), float(sample[9]),
                            float(sample[10]), float(sample[11]), float(sample[12]),
                            float(sample[13]), float(sample[14])])
            i += 1
    data = np.array(data)
    return data


if __name__ == '__main__':
    UI_filename = ""
    bp_ga = None
    my_svr = None
    lstm = None

    def open_my():
        btn1['state'] = 'disabled'
        btn2['state'] = 'disabled'
        btn3['state'] = 'disabled'
        filename = filedialog.askopenfilename()
        filename = str(filename)
        if filename != "" and filename.endswith('.csv'):
            global UI_filename
            UI_filename = filename
            try:

                btn1['state'] = 'normal'
                btn2['state'] = 'normal'
                btn3['state'] = 'normal'

            except EOFError as e:
                print(e)
        else:
            pass


    def train_my():
        global bp_ga, my_svr, lstm
        if UI_filename != "":
            try:
                # ----------BP_GA---------- #
                data = load_data(filename=UI_filename)
                train_size = int(0.7 * len(data))
                column1 = data[:train_size, 3:14]
                column2 = data[:train_size, 0:3]
                column1_v = data[train_size:, 3:14]
                column2_v = data[train_size:, 0:3]
                bp_ga = GA(20, 20, 138, 0.75, 0.05, column1, column2, column1_v, column2_v)
                bp_ga.run()

                # ----------SVR----------#
                my_svr = My_SVR(data, 6)
                my_svr.train()
                # ----------LSTM---------#
                lstm = My_LSTM(rnn_unit=10, input_size=11, time_step=20, output_size=3, lr=0.006, data=data)
                lstm.train_lstm()

            except Exception as e:
                print(e)


    def verify_my():
        global bp_ga
        # ------bp_ga---------#
        bp_ga.verify()
        # ------SVM-----------#
        my_svr.verify()
        # ------LSTM----------#
        lstm.verify()

    def getcurrent_x():
        current = [[145, 132, 139, 134, 102, 108, 144, 116, 137, 55, 33]]
        data_ = np.array(current)
        return data_


    def predict_my():
        global bp_ga
        # -------bp_ga--------#
        predict_y_bp = bp_ga.predict(getcurrent_x())
        # -------SVM----------#
        predictt_y_svr = my_svr.predict(getcurrent_x())
        # --------LSTM---------#
        predict_x = [[170.0, 168.0, 167.0, 151.0, 139.0, 174.0, 136.0, 120.0, 153.0, 61.0, 33.0, 119.0, 141.0, 148.0],
                     [168.0, 167.0, 151.0, 139.0, 174.0, 119.0, 120.0, 153.0, 133.0, 61.0, 33.0, 141.0, 148.0, 107.0],
                     [167.0, 151.0, 139.0, 174.0, 119.0, 141.0, 153.0, 133.0, 127.0, 61.0, 33.0, 148.0, 107.0, 146.0],
                     [151.0, 139.0, 174.0, 119.0, 141.0, 148.0, 133.0, 127.0, 129.0, 61.0, 33.0, 107.0, 146.0, 130.0],
                     [139.0, 174.0, 119.0, 141.0, 148.0, 107.0, 127.0, 129.0, 146.0, 61.0, 33.0, 146.0, 130.0, 116.0],
                     [174.0, 119.0, 141.0, 148.0, 107.0, 146.0, 129.0, 146.0, 130.0, 61.0, 33.0, 130.0, 116.0, 126.0],
                     [119.0, 141.0, 148.0, 107.0, 146.0, 130.0, 146.0, 130.0, 105.0, 61.0, 33.0, 116.0, 126.0, 115.0],
                     [141.0, 148.0, 107.0, 146.0, 130.0, 116.0, 130.0, 105.0, 119.0, 61.0, 33.0, 126.0, 115.0, 93.0],
                     [148.0, 107.0, 146.0, 130.0, 116.0, 126.0, 105.0, 119.0, 120.0, 61.0, 33.0, 115.0, 93.0, 119.0],
                     [107.0, 146.0, 130.0, 116.0, 126.0, 115.0, 119.0, 120.0, 99.0, 61.0, 33.0, 93.0, 119.0, 103.0],
                     [146.0, 130.0, 116.0, 126.0, 115.0, 93.0, 120.0, 99.0, 95.0, 61.0, 33.0, 119.0, 103.0, 106.0],
                     [130.0, 116.0, 126.0, 115.0, 93.0, 119.0, 99.0, 95.0, 105.0, 61.0, 33.0, 103.0, 106.0, 104.0],
                     [116.0, 126.0, 115.0, 93.0, 119.0, 103.0, 95.0, 105.0, 100.0, 61.0, 33.0, 106.0, 104.0, 99.0],
                     [126.0, 115.0, 93.0, 119.0, 103.0, 106.0, 105.0, 100.0, 92.0, 61.0, 33.0, 104.0, 99.0, 107.0],
                     [115.0, 93.0, 119.0, 103.0, 106.0, 104.0, 100.0, 92.0, 91.0, 61.0, 33.0, 99.0, 107.0, 94.0],
                     [93.0, 119.0, 103.0, 106.0, 104.0, 99.0, 92.0, 91.0, 85.0, 61.0, 33.0, 107.0, 94.0, 88.0],
                     [119.0, 103.0, 106.0, 104.0, 99.0, 107.0, 91.0, 85.0, 84.0, 61.0, 33.0, 94.0, 88.0, 81.0],
                     [103.0, 106.0, 104.0, 99.0, 107.0, 94.0, 85.0, 84.0, 79.0, 61.0, 33.0, 88.0, 81.0, 117.0],
                     [106.0, 104.0, 99.0, 107.0, 94.0, 88.0, 84.0, 79.0, 80.0, 61.0, 33.0, 81.0, 117.0, 94.0],
                     [104.0, 99.0, 107.0, 94.0, 88.0, 81.0, 79.0, 80.0, 73.0, 11.0, 33.0, 117.0, 94.0, 71.0]]
        predict_x = np.array(predict_x)
        predict_y_lstm = lstm.prediction(predict_x[:, :11])

        y1 = 0.5*predict_y_bp[0, 0] + 0.3*predict_y_lstm[0, 0] + 0.2*predictt_y_svr[0.0]
        y2 = 0.6*predict_y_bp[0, 1] + 0.4*predict_y_lstm[0, 1]
        y3 = 0.6*predict_y_bp[0, 2] + 0.4 * predict_y_lstm[0, 2]
        lab_5_['text'] = str(int(y1))+" 辆"
        lab_10_['text'] = str(int(y2))+" 辆"
        lab_15_['text'] = str(int(y3))+" 辆"


    root = Tk()

    root.title('流量训练与预测')

    left = Frame(root)

    btn = Button(left, text="choice file", state='normal', command=open_my, width=5)
    btn.pack(side=TOP, anchor=W, fill=X, expand=YES, padx=10, pady=10)

    btn1 = Button(left, text='train', state='disabled', command=train_my, width=5)
    btn1.pack(side=TOP, anchor=W, fill=X, expand=YES, padx=10, pady=10)

    btn2 = Button(left, text='verify', state='disabled', command=verify_my, width=5)
    btn2.pack(side=TOP, anchor=W, fill=X, expand=YES, padx=10, pady=10)

    btn3 = Button(left, text='predict', state='disabled', command=predict_my, width=5)
    btn3.pack(side=TOP, anchor=W, fill=X, expand=YES, padx=10, pady=10)

    left.pack(side=LEFT, fill=BOTH, expand=YES)

    right = Frame(root)

    imgInfo = PhotoImage(file="ui/login.png")

    lab = Label(right, image=imgInfo)

    lab.pack(fill=BOTH, expand=YES, side=TOP)

    rightBottom = Frame(right)

    lab_notification = Label(rightBottom, text="result:->>")
    lab_notification.pack(side=LEFT, anchor=CENTER)
    lab_5 = Label(rightBottom, text='5 min:')
    lab_5.pack(side=LEFT, anchor=CENTER, padx=15, pady=5)
    lab_5_ = Label(rightBottom, text='---  辆')
    lab_5_.pack(side=LEFT, anchor=CENTER)

    lab_10 = Label(rightBottom, text='10 min:')
    lab_10.pack(side=LEFT, anchor=CENTER, padx=15, pady=5)
    lab_10_ = Label(rightBottom, text="---  辆")
    lab_10_.pack(side=LEFT, anchor=CENTER)

    lab_15 = Label(rightBottom, text='15 min:')
    lab_15.pack(side=LEFT, anchor=CENTER, padx=15, pady=5)
    lab_15_ = Label(rightBottom, text="---  辆")
    lab_15_.pack(side=LEFT, anchor=CENTER)

    rightBottom.pack(fill=BOTH, expand=YES, side=BOTTOM)

    right.pack(side=LEFT, fill=BOTH, expand=YES)

    root.mainloop()
