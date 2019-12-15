from numpy import ndarray as nd


def to_index_list (mat, val_idx,train_or_val, set_num):
    if train_or_val == 'train' :
        l_p1 = []
        l_p2 = []
        for i in range(len(mat)):
            l1 = [[[i, k, j] for j in range(5) if
                   mat[i, 0][k, j].size != 0 and not ([i, k] in nd.tolist(val_idx[set_num][0] - 1))] for k in
                  range(len(mat[i, 0]))]
            l2 = [[[i, k, j] for j in range(5, 10) if
                   (mat[i, 0][k, j].size != 0 and not ([i, k] in nd.tolist(val_idx[set_num][0] - 1)))] for k in
                  range(len(mat[i, 0]))]
            l_p1 = l_p1 + [x for x in l1 if x]
            l_p2 = l_p2 + [x for x in l2 if x]

        return [l_p1, l_p2]
    else:
        '''
        labels = [[[val_idx[i][0][k][0], val_idx[1][0][k][1], j] for k in range(100) for j in range(10) if
                   mat[val_idx[i][0][k][0],0][ val_idx[1][0][k][1], j].size != 0] for i in range(len(val_idx))]
        '''
        val_1 =[]
        val_2 = []

        for k in range(100):
            a = []
            b = []
            for j in range(5):
                if mat[val_idx[set_num][0][k][0]-1, 0][val_idx[1][0][k][1]-1, j].size != 0:
                    a.append([val_idx[set_num][0][k][0], val_idx[set_num][0][k][1], j])
                if mat[val_idx[set_num][0][k][0]-1, 0][val_idx[1][0][k][1]-1, j+5].size != 0:
                    b.append([val_idx[set_num][0][k][0], val_idx[set_num][0][k][1], j+5])
            val_1.append(a)
            val_2.append(b)



        return [val_1, val_2]


