import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import math 
import scipy 
import cp_hw2 as helper



def convert(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def readimage(path):
    return cv2.imread(path).astype(np.float32) / 255.0

def showimage(image):
    plt.imshow(convert(image))
    plt.show()

def writeimage(file, image):
    return cv2.imwrite('./data/output/' + file, (np.clip(image,0,1) * 255).astype(np.uint8))

def subslice(I, x, y):
    return I[x::16, y::16]

def linearize(C):
    return (C < 0.0404482)/12.92 + (C > 0.0404482) * np.power((C+0.055)/1.055, 2.4)    



def make_mosaic(I):
    v_list = list()
    for i in range(16):
        h_list = list()
        for j in range(16):
            h_list.append(subslice(I, i, j))
        v_list.append(cv2.hconcat(h_list))
    return cv2.vconcat(v_list)

def get_5D(I):
    v_list = list()
    for i in range(16):
        h_list = list()
        for j in range(16):
            h_list.append(subslice(I, i, j))
        v_list.append(np.array(h_list))
    return np.array(v_list)

# L : the 5D array
# d : the the depth
# r : aperture inroad size [0,1,2,3,4,5,6,7]
def sub(L, d, r):
    lensletSize = 16
    maxUV = (lensletSize - 1) / 2; 
    U = np.arange(lensletSize) - maxUV 
    V = np.arange(lensletSize) - maxUV 
    c_list = list()
    for c in range(3):
        I = np.zeros((L.shape[2],L.shape[3]),  dtype=np.float32)
        for i in range(r, 16 - r):
            du = d * U[i]
            for j in range(r, 16 - r):
                dv = d * V[j]
                S = L[i,j,:,:,c]
                s = np.arange(0, L.shape[2])
                t = np.arange(0, L.shape[3])
                f = scipy.interpolate.interp2d(t, s, S)
                I = I + f(t - dv, s + du)
        I /= (16 - 2*r) * (16 - 2*r)
        c_list.append(I)
    res = np.dstack((c_list[0], c_list[1], c_list[2]))
    res = res.astype(np.float32)
    return res

def refocus(L, shifts):

    c_list = list()
    for c in range(3):
        I = np.zeros((1080,1920),  dtype=np.float32)
        cnt = 0
        for shift in shifts:
            du = shift[0]
            dv = shift[1]
            
            S = L[cnt,:,:,c]
            s = np.arange(0, 1080)
            t = np.arange(0, 1920)
            f = scipy.interpolate.interp2d(t, s, S)
            I = I + f(t + dv, s + du)
            cnt+=1
        I /= cnt
        c_list.append(I)
    res = np.dstack((c_list[0], c_list[1], c_list[2]))
    res = res.astype(np.float32)
    return res


def confocal(L):
    dstack = np.arange(0, 1.6, 0.2)
    rstack = np.arange(0, 8, 1)

    # 2D mosaic
    # v_list = list()
    # for d in dstack:
    #     h_list = list()
    #     for r in rstack:
    #         h_list.append(sub(L, d, r))
    #     v_list.append(cv2.hconcat(h_list))
    # return cv2.vconcat(v_list)

    # compute the array
    # v_list = list()
    # for d in dstack:
    #     h_list = list()
    #     for r in rstack:
    #         h_list.append(sub(L, d, r))
    #     v_list.append(np.array(h_list))
    # output = np.array(v_list)

    # with open('data/2Darray.npy', 'wb') as f:
    #     np.save(f, output)
    # return output




    # load the array 
    input = np.zeros((1,1))
    with open('data/2Darray.npy', 'rb') as f:
        input = np.load(f)      

    dim = (input.shape[2],input.shape[3])
    d_map = np.zeros(dim)
    eps = 1e-06

    inf = 1e9
    Vmin = np.ones(dim) * inf
    for d in range(8):
        V = np.zeros(dim)
        for c in range(3):
            S = np.zeros(dim)
            for r in range(8):
                I = input[d, r, :, :, c]
                S += I
            S /= 8
            for r in range(8):
                I = input[d, r, :, :, c]
                V += (I-S)*(I-S)
        Vmin = np.minimum(Vmin, V)
    for d in range(8):
        V = np.zeros(dim)
        for c in range(3):
            S = np.zeros(dim)
            for r in range(8):
                I = input[d, r, :, :, c]
                S += I
            S /= 8
            for r in range(8):
                I = input[d, r, :, :, c]
                V += (I-S)*(I-S)
        d_map += ((np.abs(V-Vmin)) < eps) * dstack[d]

    
    d_map = np.dstack([d_map,d_map,d_map])
    d_map = rescale(1 - d_map * 0.75)
    d_map = d_map.astype(np.float32)
    return d_map

    # print(d_map)
    # plt.imshow(d_map)
    # plt.show()

    # pixels = [[100,100],[214,369], [337,608]]
    # for p in pixels:
    #     o = input[:, :, p[0], p[1]]
    #     data = np.mean(o, axis=2)
    #     x = np.arange(0, 1.6, 0.2)
    #     y = np.arange(16, 0, -2)
    #     plt.title("plot for pixel({:d},{:d})".format(p[0],p[1]))
    #     plt.imshow(
    #         data, cmap='gray', interpolation='nearest', origin='lower')
    #     plt.xlabel("focus depth")
    #     plt.ylabel("aperture radius(square)")

    #     x_list = list()
    #     for e in np.arange(0 - 0.2, 1.6+0.2, 0.2):
    #         x_list.append('{:.1f}'.format(e))
    #     y_list = list()
    #     for e in np.arange(16+2, 0-2, -2):
    #         y_list.append('{:.1f}'.format(e))
    #     plt.xticks(plt.xticks()[0], x_list)
    #     plt.yticks(plt.yticks()[0], y_list)
    #     plt.savefig("data/output/pixel({:d},{:d}).png".format(p[0],p[1]))
        #plt.show()
                
def gen_stack():
    local_stack = list()
    local_stack.append(readimage("data/output/sub0.png"));
    local_stack.append(readimage("data/output/sub0.25.png"));
    local_stack.append(readimage("data/output/sub0.5.png"));
    local_stack.append(readimage("data/output/sub0.75.png"));
    local_stack.append(readimage("data/output/sub1.png"));
    local_stack.append(readimage("data/output/sub1.25.png"));
    return local_stack



def all_in(stack, r, sigma_1, sigma_2):
    depths = [0,0.25,0.5,0.75,1,1.25]
    dim = (stack[0].shape[0], stack[0].shape[1], 3)
    P = np.zeros(dim)
    Q = np.zeros(dim)
    dP = np.zeros((stack[0].shape[0], stack[0].shape[1]))
    dQ = np.zeros((stack[0].shape[0], stack[0].shape[1]))
    for i in range(6):
        I = stack[i]
        d = depths[i]
        xyz = helper.lRGB2XYZ(linearize(I))
        L = xyz[:,:,1]

        low = cv2.GaussianBlur(L, (r, r), sigma_1)
        high = L - low 
        w = cv2.GaussianBlur(high * high, (r, r), sigma_2)
        W = np.dstack((w,w,w))
        P += W * I 
        Q += W 
        dP += w * d
        dQ += w
    return (P/Q, dP/dQ)


def rescale(I):
    return I/(np.max(I))

def gen_frames():
    cap = cv2.VideoCapture("data/input/bear.MOV")
    count = 0
    while cap.isOpened():
        ret,frame = cap.read()
        if ret == False:
            continue
        # cv2.imshow('window-name', frame)
        # cv2.imwrite("frame%d.jpg" % count, frame)
        count = count + 1
        if count % 68 == 0:
            cv2.imwrite("data/input/frame{:d}.jpg".format(count//68), frame)
        print(count)
    print("total count : {:d}".format(count))
    print(V)
    I = readimage("data/input/DSC_0106.JPG")
    plt.imshow(convert(I))
    plt.show()

# template and the frame match
def match(T, I):
    def cut(J, i, j, R):
        return J[i:i+R,j:j+R]
    # keep the original
    ori_T = T
    ori_I = I
    # the luminance channel
    T = helper.lRGB2XYZ(linearize(T))[:,:,1]
    I = helper.lRGB2XYZ(linearize(I))[:,:,1]


    # template left top
    o = [570,1412]
    # template radius
    r = 50
    # window radius
    w_r = 500
    g = cut(T, o[0], o[1], r)
    box = np.ones((r,r)) / (r*r)
    I_bar = scipy.signal.convolve2d(I, box, mode = 'same')
    g_diff = g - np.mean(g)

    corr = -1e9
    target_i = 0
    target_j = 0
    for i in range(o[0] - 400, o[0] + 300):
        for j in range(o[1] - 400, o[1] + 400):
            I_sub = cut(I, i, j, r)
            # print("index : {:d},{:d}".format(i,j))
            # assert(I_sub.shape[0] == r)
            #print(I_sub.shape)
            I_b = I_bar[i,j]
            P = np.sum((g_diff) * (I_sub - I_b))
            left = np.sum(g_diff * g_diff)
            right = np.sum(I_sub * I_sub) + (I_b * I_b * r * r) - np.sum(2 * I_sub * I_b)
            Q = np.sqrt(left) * np.sqrt(right)    
            h = P/Q
            if h > corr:
                corr = h
                target_i = i 
                target_j = j

    # print(target_i, target_j)
    # L = cut(ori_T, o[0], o[1], r)
    # R = cut(ori_I, target_i, target_j, r)
    # plt.imshow(convert(cv2.hconcat([L,R])))
    # plt.show()
    # plt.imshow(convert(ori_I))
    # plt.show()
    return (target_i - o[0], target_j - o[1])


def main():
    # part 1 A,B
    # I = readimage("data/chessboard_lightfield.png")
    # L = get_5D(I)
    # writeimage('sub0.25.png', sub(L,0.25))
    # writeimage('sub0.5.png', sub(L,0.5))
    # writeimage('sub0.75.png', sub(L,0.75))
    # writeimage('sub1.25.png', sub(L,1.25))

    # writeimage('stack.png', make_mosaic(I))

    # part 1C
    # S = gen_stack()
    # I_all, map = all_in(S, 9, 16, 64)
    # map = 1 - rescale(map)
    # writeimage("I_all.png", I_all)
    # writeimage("I_all_map.png", map)

    # part 1D
    # res = confocal(L)
    # writeimage('dmap.png', res)

    # part 3
    # T = readimage("data/input/frame16.jpg")
    # plt.imshow(convert(T))
    # plt.show()

    # T = readimage("data/input/frame16.jpg")
    # shift = list()
    # for i in range(12, 22):        
    #     I = readimage("data/input/frame{:d}.jpg".format(i))
    #     print(i)
    #     shift.append(match(T,I))
    # with open('data/shifts_3.npy', 'wb') as f:
    #     np.save(f, np.array(shift))

    with open('data/shifts_3.npy', 'rb') as f:
        shifts = np.load(f)
    L = list()
    for i in range(12, 22): 
        I = readimage("data/input/frame{:d}.jpg".format(i))
        L.append(I)
    L = np.array(L)
    eyes = refocus(L,shifts)
    writeimage("plug.jpg", eyes)

    # fig, ax = plt.subplots()
    # ax.imshow(convert(T))
    # rect = patches.Rectangle((900, 700), 30, 30, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    # plt.show()

    


main()