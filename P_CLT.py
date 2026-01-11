import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
matplotlib.use('Agg')  # 在import pyplot前设置
import networkx as nx
import numpy as np
import math
from matplotlib.font_manager import FontProperties

seed1 = 89684

g=0.1
nodes_num = 20
distance_node = 0.6
G = nx.random_geometric_graph(nodes_num, 0, seed=seed1)  # ,,seed=89680
added_user_num = 0
step = 1
scale = 1

# 如果有新的节点，合并到原图中
if added_user_num != 0:
    H = nx.random_geometric_graph(step * added_user_num, 0, seed=seed1 + 10)
    mapping = dict(zip(H, range(nodes_num, nodes_num + step * added_user_num)))
    H = nx.relabel_nodes(H, mapping)
    U = nx.compose(G, H)
else:
    U = G
# 计算节点之间的距离，确定是否添加边
pos = nx.get_node_attributes(U, "pos")
#
pos[18][1] = 0.65
pos[18][0] = 0.5

#
# pos[18][1] = 0.6
# pos[18][0] = 0.45
for nodeG in range(nodes_num + step * added_user_num):
    for nodeH in range(nodes_num + step * added_user_num):
        if nodeH != nodeG:
            distance = [pos[nodeG][0] - pos[nodeH][0], pos[nodeG][1] - pos[nodeH][1]]
            d = math.hypot(distance[0], distance[1])
            if d < distance_node:
                U.add_edge(nodeG, nodeH)
# 计算节点间的速率率，并更新权重
nodes_num = U.number_of_nodes()
one_hop_error = np.zeros((nodes_num, nodes_num))
for node1 in range(nodes_num):
    for node2 in range(nodes_num):
        if U.has_edge(node1, node2):
            distance = [pos[node1][0] - pos[node2][0], pos[node1][1] - pos[node2][1]]
            d = math.hypot(distance[0], distance[1])
            d = d * scale
            P = 78
            B = 30
            f = 2500
            N = 56
            lamda = 41.7
            np.random.seed(4)
            h = (lamda / (d * f))
            snr = (h) ** 2 * P / N * B

            error = 1 / (B * math.log2(1 + snr))

            one_hop_error[node1][node2] = error  #
            Weight = -error
            # print(error)
            U[node1][node2].update({"weight": Weight})

# 将没有边连接的节点权重设为无穷大
for i in range(nodes_num):
    for j in range(nodes_num):
        if one_hop_error[i][j] == 0 and i != j:
            one_hop_error[i][j] = float('inf')

edges = []
for i in range(nodes_num):
    for j in range(nodes_num):
        if one_hop_error[i][j] != float('inf') and one_hop_error[i][j] != 0:
            edges.append([i, j, one_hop_error[i][j]])

# 按权重从小到大进行排序
edges = sorted(edges, key=lambda item: item[2])
print(len(edges))

for i in range(nodes_num):
    for j in range(nodes_num):
        if one_hop_error[i][j] == float('inf'):
            one_hop_error[i][j] = -1
        # Kruskal算法计算最小生成树
flags = []
for l in range(nodes_num):
    flags.append(l)
index = 0
r = 0
while index < len(edges):
    ivertex = edges[index][0]
    jvertex = edges[index][1]
    if flags[ivertex] != flags[jvertex]:
        # 两个顶点不属于同一连通分量
        # 找到它们各自的连通分量的序号
        iflag = flags[ivertex]
        jflag = flags[jvertex]
        for limit in range(nodes_num):
            if flags[limit] == jflag:
                # 将j和i的连通序号设置相同, 表示它俩是连通的
                flags[limit] = iflag

        index += 1

    else:
        # 已经连通了, 即加入这条边就构成了环
        # 删除这条边
        edges.pop(index)

print(edges)

# 用于记录最小生成树每个节点的分支数，即节点向其他节点广播模型的邻接节点个数。
# 构建最小生成树的邻接矩阵
Ve = np.zeros((nodes_num, nodes_num))
for i in range(len(edges)):
    Ve[edges[i][0]][edges[i][1]] = edges[i][2]
    Ve[edges[i][1]][edges[i][0]] = edges[i][2]

for i in range(nodes_num):
    for j in range(nodes_num):
        if Ve[i][j] == 0:
            Ve[i][j] = -1

Ve = list(Ve)
Ve = np.array(Ve)
ve = [Ve for i in range(nodes_num)]
ve = list(ve)
ve = np.array(ve)


def in_same_interval(speed1, speed2):
    """
    判断两个链路传输速率是否处于同一区间。

    规则：




    1. 如果链路传输速率小于 0.1，前后差异小于 0.05 则为同一区间。
    2. 如果链路传输速率大于或等于 0.1，前后差异小于 0.3 则为同一区间。
    """
    if speed1 < 0.1 and speed2 < 0.1:
        return abs(speed1 - speed2) < g
    elif speed1 >= 0.1 and speed2 >= 0.1:
        return abs(speed1 - speed2) < g
    elif speed1 < 0.1 and speed2 >= 0.1:
        return abs(speed1 - speed2) < g
    elif speed1 >= 0.1 and speed1 < 0.1:
        return abs(speed1 - speed2) < g


obs_n = [[i] for i in range(nodes_num)]
end_node_n = [i for i in range(nodes_num)]
L_A = [0 for j in range(nodes_num)]

for k in range(nodes_num):
    a = np.where(Ve[k, :] > 0)[0]
    L_A[k] = len(a)

L = [L_A for j in range(nodes_num)]

short_path = [[] for i in range(nodes_num)]
# for k in range(len(obs_n)):
#
#     memory = [[i] for i in range(nodes_num)]
#     object_n = [[i] for i in range(nodes_num)]
#     new_obs_n = []
#     while object_n[k] != end_node_n:
#         for i in range(len(obs_n[k])):  # 遍历第k个客户端的i个节点
#             c = []
#             max = []
#             b = []
#             possible_actions = np.where(ve[k][obs_n[k][i], :] > 0)[0]
#             for x, n in enumerate(possible_actions):
#                 if n not in memory[k]:
#                     b.append(n)
#                     c.append(n)
#             action = b
#             L[k][obs_n[k][i]] = len(action)
#
#             for j in range(len(action)):
#                 new_obs_n.append(action[j])
#                 memory[k].append(action[j])
#                 max.append(ve[k][obs_n[k][i], action[j]])
#                 object_n[k].append(action[j])
#                 short_path[k].append([obs_n[k][i], action[j]])
#         object_n[k].sort()
#         obs_n[k] = new_obs_n
#         new_obs_n = []
#
#


obs_n = [[i] for i in range(nodes_num)]
end_node_n = [i for i in range(nodes_num)]

r = 0
R = []

path = []

lst = list(pos.values())

short_path = [[] for i in range(nodes_num)]
for k in range(len(obs_n)):

    memory = [[i] for i in range(nodes_num)]
    object_n = [[i] for i in range(nodes_num)]

    new_obs_n = []

    while object_n[k] != end_node_n:

        for i in range(len(obs_n[k])):  # 遍历第k个客户端的i个节点
            c = []
            max = []
            b = []
            possible_actions = np.where(ve[k][obs_n[k][i], :] > 0)[0]
            for x, n in enumerate(possible_actions):
                if n not in memory[k]:
                    b.append(n)
                    c.append(n)
            action = b

            for j in range(len(action)):
                new_obs_n.append(action[j])
                memory[k].append(action[j])
                max.append(ve[k][obs_n[k][i]][action[j]])
                object_n[k].append(action[j])
                short_path[k].append([obs_n[k][i], action[j]])

            max.sort()
            if max != []:
                r += max[len(action) - 1]
            object_n[k].sort()

        obs_n[k] = new_obs_n
        new_obs_n = []

R.append(r)

# 添加邻接节点中链路速率在同一传输区间的链路（同时避免环路），修改链路（减少广播次数）
memory_n = [[] for i in range(nodes_num)]
obss_n = [[i] for i in range(nodes_num)]
end_node_n = [i for i in range(nodes_num)]
#
for i in range(len(obss_n)):

    new_obss_n = []
    while obss_n[i] != []:
        for j in range(len(obss_n[i])):
            next_memory = []
            if obss_n[i][j] not in memory_n[i]:
                memory_n[i].append(obss_n[i][j])

            mst_neighborss = np.where(ve[i][obss_n[i][j], :] > 0)[0]
            # obs_n[i] = mst_neighborss

            for node in mst_neighborss:
                next_memory.append(node)
                if node in memory_n[i]:
                    # ve[k][node,obss_n[i][j]] = -1
                    # ve[k][obss_n[i][j], node] = -1
                    mst_neighborss = mst_neighborss[mst_neighborss != node]
                else:
                    new_obss_n.append(node)
                    # memory_n[i].append(node)
            neighbors = np.where(one_hop_error[obss_n[i][j], :] > 0)[0]
            for node in memory_n[i]:
                neighbors = neighbors[neighbors != node]


            for neighbor in neighbors:
                edge_speed = one_hop_error[obss_n[i][j]][neighbor]
                add_link = True
                for mst_neighbor in mst_neighborss:
                    add_link = True

                    mst_speed = one_hop_error[obss_n[i][j]][mst_neighbor]

                    if not in_same_interval(edge_speed, mst_speed):
                        add_link = False

                    if add_link:
                        for p_node in memory_n[i]:
                            ve[i][neighbor, p_node] = -1
                            ve[i][p_node, neighbor] = -1
                        # pa=np.where(ve[i][neighbor, :] > 0)[0]

                        ve[i][obss_n[i][j], neighbor] = one_hop_error[obss_n[i][j]][neighbor]
                        ve[i][neighbor, obss_n[i][j]] = one_hop_error[obss_n[i][j]][neighbor]

                        if neighbor not in memory_n[i]:
                            memory_n[i].append(neighbor)
                            new_obss_n.append(neighbor)

        new_obss_n = sorted(new_obss_n, key=lambda n: L[i][n], reverse=True)

        obss_n[i] = new_obss_n

        new_obss_n = []
        memory_n[i].sort()

# 根据生成树来广播
obs_n = [[i] for i in range(nodes_num)]
end_node_n = [i for i in range(nodes_num)]

r = 0
# R = []

path = []

lst = list(pos.values())

short_path = [[] for i in range(nodes_num)]
for k in range(len(obs_n)):

    memory = [[i] for i in range(nodes_num)]
    object_n = [[i] for i in range(nodes_num)]

    new_obs_n = []

    while object_n[k] != end_node_n:

        for i in range(len(obs_n[k])):  # 遍历第k个客户端的i个节点
            c = []
            max = []
            b = []
            possible_actions = np.where(ve[k][obs_n[k][i], :] > 0)[0]
            for x, n in enumerate(possible_actions):
                if n not in memory[k]:
                    b.append(n)
                    c.append(n)
            action = b

            for j in range(len(action)):
                new_obs_n.append(action[j])
                memory[k].append(action[j])
                max.append(ve[k][obs_n[k][i]][action[j]])
                object_n[k].append(action[j])
                short_path[k].append([obs_n[k][i], action[j]])

            max.sort()
            if max != []:
                r += max[len(action) - 1]
            object_n[k].sort()

        obs_n[k] = new_obs_n
        new_obs_n = []

R.append(r)

for e in range(5):

    obs_n = [[i] for i in range(nodes_num)]
    end_node_n = [i for i in range(nodes_num)]
    L = [[0 for i in range(nodes_num)] for j in range(nodes_num)]

    for k in range(nodes_num):
        for l in range(nodes_num):
            a = np.where(ve[k][l, :] > 0)[0]
            L[k][l] = len(a)


    memory_n = [[] for i in range(nodes_num)]
    obss_n = [[i] for i in range(nodes_num)]
    end_node_n = [i for i in range(nodes_num)]

    for i in range(len(obss_n)):

        new_obss_n = []
        while obss_n[i] != []:
            for j in range(len(obss_n[i])):
                next_memory = []
                if obss_n[i][j] not in memory_n[i]:
                    memory_n[i].append(obss_n[i][j])

                mst_neighborss = np.where(ve[i][obss_n[i][j], :] > 0)[0]
                # obs_n[i] = mst_neighborss

                for node in mst_neighborss:
                    next_memory.append(node)
                    if node in memory_n[i]:
                        # ve[k][node,obss_n[i][j]] = -1
                        # ve[k][obss_n[i][j], node] = -1
                        mst_neighborss = mst_neighborss[mst_neighborss != node]
                    else:
                        new_obss_n.append(node)
                        # memory_n[i].append(node)
                neighbors = np.where(one_hop_error[obss_n[i][j], :] > 0)[0]
                for node in memory_n[i]:
                    neighbors = neighbors[neighbors != node]


                for neighbor in neighbors:
                    edge_speed = one_hop_error[obss_n[i][j]][neighbor]

                    for mst_neighbor in mst_neighborss:

                        mst_speed = one_hop_error[obss_n[i][j]][mst_neighbor]

                        mst_interval = []
                        for mst_neighbor in mst_neighborss:
                            mst_speed = one_hop_error[obss_n[i][j]][mst_neighbor]
                            mst_interval.append(mst_speed)
                        mst_interval.sort()

                        if edge_speed < mst_interval[-1]:
                            add_link = True
                        else:
                            add_link = False

                        # 如果该链路与其他链路在同一区间，加入MST
                        if add_link:
                            for p_node in memory_n[i]:
                                ve[i][neighbor, p_node] = -1
                                ve[i][p_node, neighbor] = -1
                            # pa=np.where(ve[i][neighbor, :] > 0)[0]

                            ve[i][obss_n[i][j], neighbor] = one_hop_error[obss_n[i][j]][neighbor]
                            ve[i][neighbor, obss_n[i][j]] = one_hop_error[obss_n[i][j]][neighbor]

                            if neighbor not in memory_n[i]:
                                memory_n[i].append(neighbor)
                            if neighbor not in memory_n[i] and neighbor not in new_obss_n:
                                new_obss_n.append(neighbor)

            new_obss_n = sorted(new_obss_n, key=lambda n: L[i][n], reverse=True)

            obss_n[i] = new_obss_n

            new_obss_n = []
            memory_n[i].sort()

    # 根据生成树来广播

    obs_n = [[i] for i in range(nodes_num)]
    end_node_n = [i for i in range(nodes_num)]
    r = 0
    path = []
    lst = list(pos.values())
    short_path = [[] for i in range(nodes_num)]
    for k in range(len(obs_n)):

        memory = [[i] for i in range(nodes_num)]
        object_n = [[i] for i in range(nodes_num)]

        new_obs_n = []

        while object_n[k] != end_node_n:

            for i in range(len(obs_n[k])):  # 遍历第k个客户端的i个节点
                c = []
                max_n = []
                b = []
                possible_actions = np.where(ve[k][obs_n[k][i], :] > 0)[0]
                for x, n in enumerate(possible_actions):
                    if n not in memory[k]:
                        b.append(n)
                        c.append(n)
                action = b

                for j in range(len(action)):
                    new_obs_n.append(action[j])
                    memory[k].append(action[j])
                    max_n.append(ve[k][obs_n[k][i]][action[j]])
                    object_n[k].append(action[j])
                    short_path[k].append([obs_n[k][i], action[j]])

                max_n.sort()
                if max_n != []:
                    r += max_n[len(action) - 1]

                object_n[k].sort()

            obs_n[k] = new_obs_n
            new_obs_n = []

    R.append(r)



#根据生成树来广播
obs_n=[[i] for i in range(nodes_num)]
end_node_n=[i for i in range(nodes_num)]



path=[]


lst = list(pos.values())

# short_path = [[] for i in range(nodes_num)]
# # 节点位置数据
# pos = {0: [0.4105802402300718, 0.2526295598729481], 1: [0.6799762057668526, 0.5846197320405553],
#        2: [0.493957098692813, 0.053151611653984854], 3: [0.3523172878612463, 0.1586292351751868],
#        4: [0.9821508359779506, 0.11148899461025719], 5: [0.32342589300772573, 0.3595049433965928],
#        6: [0.6677160663204655, 0.4988550013579629], 7: [0.8498537506051752, 0.99562290855221],
#        8: [0.8090578447812037, 0.7479776066059564], 9: [0.4737923860499168, 0.9666339132889018],
#        10: [0.03234124380779957, 0.10749448061324862], 11: [0.38332390417784123, 0.6871359384397971],
#        12: [0.3812407857059459, 0.06942283101533808], 13: [0.44893036940387576, 0.04474708677027184],
#        14: [0.8604584782809026, 0.4871470573203861], 15: [0.00622200530673922, 0.44581576071945705],
#        16: [0.6859581560236029, 0.8828417884984188], 17: [0.8714615725241548, 0.04855864703255497], 18: [0.5, 0.65],
#        19: [0.04074764574267342, 0.24783419134788431]}
#
# # 边列表（带权重）
# edges = [[2, 13, 0.01230211811778686], [13, 2, 0.01230211811778686], [12, 13, 0.01965575823847005],
#          [13, 12, 0.01965575823847005], [1, 6, 0.024693878806829977], [6, 1, 0.024693878806829977],
#          [3, 12, 0.027427190658685], [12, 3, 0.027427190658685], [0, 3, 0.034582646642443635],
#          [3, 0, 0.034582646642443635], [2, 12, 0.0361045901877062], [12, 2, 0.0361045901877062],
#          [11, 18, 0.040249002531749366], [18, 11, 0.040249002531749366], [4, 17, 0.04273688565190675],
#          [17, 4, 0.04273688565190675], [0, 5, 0.048434164696737725], [5, 0, 0.048434164696737725],
#          [10, 19, 0.04994761539790772], [19, 10, 0.04994761539790772], [3, 13, 0.05507237172871409],
#          [13, 3, 0.05507237172871409], [2, 3, 0.07292506872417927], [3, 2, 0.07292506872417927],
#          [8, 16, 0.07724086764794179], [16, 8, 0.07724086764794179], [0, 12, 0.07941013020795504],
#          [12, 0, 0.07941013020795504], [1, 18, 0.08389271323215444], [18, 1, 0.08389271323215444],
#          [6, 14, 0.08513399083688804], [14, 6, 0.08513399083688804], [7, 16, 0.08972095707127459],
#          [16, 7, 0.08972095707127459], [15, 19, 0.09133425958966908], [19, 15, 0.09133425958966908],
#          [3, 5, 0.0929266157046185], [5, 3, 0.0929266157046185], [1, 14, 0.09470237984415517],
#          [14, 1, 0.09470237984415517], [1, 8, 0.09724476213984141], [8, 1, 0.09724476213984141],
#          [0, 13, 0.0999160751001191], [13, 0, 0.0999160751001191], [0, 2, 0.10402226868145509],
#          [2, 0, 0.10402226868145509], [6, 18, 0.112461640843615], [18, 6, 0.112461640843615],
#          [9, 16, 0.11457944097543586], [16, 9, 0.11457944097543586], [7, 8, 0.13641751273619318],
#          [8, 7, 0.13641751273619318], [8, 14, 0.15171772518237245], [14, 8, 0.15171772518237245],
#          [6, 8, 0.17434242766037134], [8, 6, 0.17434242766037134], [9, 11, 0.18282870131014195],
#          [11, 9, 0.18282870131014195], [5, 12, 0.18518990472790745], [12, 5, 0.18518990472790745],
#          [16, 18, 0.1877878507463325], [18, 16, 0.1877878507463325], [1, 16, 0.1881389573509973],
#          [16, 1, 0.1881389573509973], [5, 19, 0.1949145895824966], [19, 5, 0.1949145895824966],
#          [1, 11, 0.2071203285177668], [11, 1, 0.2071203285177668], [9, 18, 0.21195800392221542],
#          [18, 9, 0.21195800392221542], [3, 10, 0.22002569231465433], [10, 3, 0.22002569231465433],
#          [3, 19, 0.22009279712748409], [19, 3, 0.22009279712748409], [8, 18, 0.22025829369851924],
#          [18, 8, 0.22025829369851924], [5, 15, 0.22612933381233005], [15, 5, 0.22612933381233005],
#          [5, 11, 0.2318220424792517], [11, 5, 0.2318220424792517], [5, 13, 0.23956724330039894],
#          [13, 5, 0.23956724330039894], [10, 15, 0.24020294840416367], [15, 10, 0.24020294840416367],
#          [5, 18, 0.24104285060564873], [18, 5, 0.24104285060564873], [6, 11, 0.24256002940391425],
#          [11, 6, 0.24256002940391425], [2, 5, 0.25569548392443964], [5, 2, 0.25569548392443964],
#          [10, 12, 0.2561868728139975], [12, 10, 0.2561868728139975], [0, 6, 0.2632775270559503],
#          [6, 0, 0.2632775270559503], [11, 16, 0.2695267719190448], [16, 11, 0.2695267719190448],
#          [0, 19, 0.28326941463360045], [19, 0, 0.28326941463360045], [5, 6, 0.28556615594940665],
#          [6, 5, 0.28556615594940665], [7, 9, 0.29413308621610446], [9, 7, 0.29413308621610446],
#          [2, 17, 0.2946664018365316], [17, 2, 0.2946664018365316], [12, 19, 0.30507649853709984],
#          [19, 12, 0.30507649853709984], [6, 16, 0.3051012229714329], [16, 6, 0.3051012229714329],
#          [5, 10, 0.3060176894469009], [10, 5, 0.3060176894469009], [4, 14, 0.3213046214262303],
#          [14, 4, 0.3213046214262303], [14, 18, 0.3223452422325939], [18, 14, 0.3223452422325939],
#          [8, 9, 0.32982517692967667], [9, 8, 0.32982517692967667], [0, 10, 0.3376097767003414],
#          [10, 0, 0.3376097767003414], [0, 18, 0.34112928380536284], [18, 0, 0.34112928380536284],
#          [10, 13, 0.36416066075124864], [13, 10, 0.36416066075124864], [1, 5, 0.36479894483364056],
#          [5, 1, 0.36479894483364056], [13, 17, 0.3662747830761091], [17, 13, 0.3662747830761091],
#          [0, 1, 0.37471304741958283], [1, 0, 0.37471304741958283], [8, 11, 0.3790060453298667],
#          [11, 8, 0.3790060453298667], [14, 16, 0.38312834329168927], [16, 14, 0.38312834329168927],
#          [1, 9, 0.38595510973317027], [9, 1, 0.38595510973317027], [0, 11, 0.3881261862829872],
#          [11, 0, 0.3881261862829872], [14, 17, 0.3939751245887631], [17, 14, 0.3939751245887631],
#          [1, 7, 0.4045135064999749], [7, 1, 0.4045135064999749], [11, 15, 0.4097999637775665],
#          [15, 11, 0.4097999637775665], [0, 15, 0.4105657507668462], [15, 0, 0.4105657507668462],
#          [3, 15, 0.41341156179884325], [15, 3, 0.41341156179884325], [13, 19, 0.4245427296138251],
#          [19, 13, 0.4245427296138251], [3, 6, 0.4391982059561756], [6, 3, 0.4391982059561756],
#          [2, 10, 0.44081310413843655], [10, 2, 0.44081310413843655], [2, 6, 0.4662599466381377],
#          [6, 2, 0.4662599466381377], [12, 17, 0.4899305470057673], [17, 12, 0.4899305470057673],
#          [2, 4, 0.49188778001518646], [4, 2, 0.49188778001518646], [7, 18, 0.49211928725393955],
#          [18, 7, 0.49211928725393955], [2, 19, 0.49499611699163415], [19, 2, 0.49499611699163415],
#          [6, 17, 0.49694197581918914], [17, 6, 0.49694197581918914], [4, 6, 0.5061703086519644],
#          [6, 4, 0.5061703086519644], [0, 17, 0.5163771072376215], [17, 0, 0.5163771072376215],
#          [6, 13, 0.5164260835219563], [13, 6, 0.5164260835219563], [6, 9, 0.5210819790007701],
#          [9, 6, 0.5210819790007701], [0, 14, 0.5230008869006998], [14, 0, 0.5230008869006998],
#          [7, 14, 0.5255278037865495], [14, 7, 0.5255278037865495], [3, 18, 0.5346618493586442],
#          [18, 3, 0.5346618493586442], [6, 12, 0.5410715160631072], [12, 6, 0.5410715160631072],
#          [11, 14, 0.5434028432105765], [14, 11, 0.5434028432105765], [6, 7, 0.5678504074885248],
#          [7, 6, 0.5678504074885248], [3, 11, 0.5685028718546387], [11, 3, 0.5685028718546387],
#          [3, 17, 0.5711774318427046], [17, 3, 0.5711774318427046], [12, 15, 0.5725377913217787],
#          [15, 12, 0.5725377913217787], [15, 18, 0.5788927218512443], [18, 15, 0.5788927218512443],
#          [4, 13, 0.5853936234107014], [13, 4, 0.5853936234107014], [1, 3, 0.5854924528503992],
#          [3, 1, 0.5854924528503992], [5, 14, 0.6170327978476501], [14, 5, 0.6170327978476501],
#          [11, 19, 0.6282589170964151], [19, 11, 0.6282589170964151], [7, 11, 0.6331679163507724],
#          [11, 7, 0.6331679163507724], [1, 4, 0.6378346231631231], [4, 1, 0.6378346231631231],
#          [1, 2, 0.6416096326722874], [2, 1, 0.6416096326722874], [2, 14, 0.6527677315263517],
#          [14, 2, 0.6527677315263517], [1, 17, 0.6554566081758798], [17, 1, 0.6554566081758798],
#          [1, 13, 0.6968312923361163], [13, 1, 0.6968312923361163], [0, 4, 0.7003472228399779],
#          [4, 0, 0.7003472228399779], [12, 18, 0.7094105475541751], [18, 12, 0.7094105475541751],
#          [1, 12, 0.7163612723023838], [12, 1, 0.7163612723023838], [2, 18, 0.7195290908912024],
#          [18, 2, 0.7195290908912024], [13, 15, 0.7206863740777547], [15, 13, 0.7206863740777547]]
# # 创建图形
# plt.figure(figsize=(12, 10))
# k = 13
# # 1. 准备边数据
# edge_pos = np.array([(pos[u], pos[v]) for u, v, w in edges])
# edge_weights = np.array([w for _, _, w in edges])
#
# # 2. 绘制浅色边（关键修改：使用Pastel1颜色映射并提高alpha值）
# norm = plt.Normalize(edge_weights.min(), edge_weights.max())
# cmap = plt.cm.Pastel1  # 改为柔和的Pastel1颜色映射
#
# lc = LineCollection(edge_pos,
#                    linewidths=1,  # 稍微加粗线宽
#                    cmap=cmap,
#                    norm=norm,
# linestyles='--',
#                     alpha=0.5,
#                    zorder=1)
# lc.set_array(edge_weights)  # 设置颜色映射的值
# plt.gca().add_collection(lc)
#
#
# from matplotlib.font_manager import FontProperties
#
#
# # 1. 绘制普通节点（浅绿色）
# roman_font = FontProperties(family='Times New Roman', size=15)
# # 1. 绘制普通节点（浅绿色）
# plt.scatter(
#     x=[pos[i][0] for i in pos if i != 13],  # 排除第14个节点
#     y=[pos[i][1] for i in pos if i != 13],
#     s=980,
#     c='#D5E8D4',
#     edgecolors='#7F7F7F',
#     linewidths=1,
#     zorder=2,
#     label="Other Clients"
# )
# # 2. 单独绘制第14个节点（黄色作为源节点）
# plt.scatter(
#     x=[pos[k][0]],
#     y=[pos[k][1]],
#     s=850,  # 放大节点尺寸
#     c='#FFD700',  # 金黄色
#     edgecolors='#FF8C00',  # 深橙色边框
#     linewidths=1.5,  # 加粗边框
#     zorder=3,  # 确保在最上层
#     label="Source Client"
# )
#
#
# # 3. 添加图例（调整字体和样式）
# font = FontProperties()
# font.set_size(16)
# # 创建字体配置，设置加粗
# font = {'family': 'Times New Roman',
#         'size': 20,
#         'weight': 'bold'}  # 添加 weight 参数
#
# for node, (x, y) in pos.items():
#     plt.text(x, y, str(node),
#              ha='center', va='center',
#              color='#333333',  # 深灰色文字
#              fontsize=20,
#              fontweight='normal',  # 不加粗
#              zorder=3)
#
# # 7. 坐标轴设置
# plt.xticks(np.arange(0, 1.1, 0.1),fontsize=25)
# plt.yticks(np.arange(0, 1.1, 0.1),fontsize=25)
# plt.xlim(-0.05, 1.05)  # 添加5%边距
# plt.ylim(-0.05, 1.05)
# plt.grid(True, linestyle=':', alpha=0.4)
# plt.ylabel(r"$\mathregular{\times \! 10^3 \, m}$", fontsize=35)
# plt.xlabel(r"$\mathregular{\times \! 10^3 \, m}$", fontsize=35)
#
#
# # plt.title("Weighted Network Visualization", pad=20)
#
# memory = [[i] for i in range(nodes_num)]
# object_n = [[i] for i in range(nodes_num)]
#
# new_obs_n = []
#
# while object_n[k] != end_node_n:
#
#     for i in range(len(obs_n[k])):  # 遍历第k个客户端的i个节点
#         c = []
#         max = []
#         b = []
#         possible_actions = np.where(ve[k][obs_n[k][i], :] > 0)[0]
#         for x, n in enumerate(possible_actions):
#             if n not in memory[k]:
#                 b.append(n)
#                 c.append(n)
#         action = b
#
#         for j in range(len(action)):
#             new_obs_n.append(action[j])
#             memory[k].append(action[j])
#             max.append(ve[k][obs_n[k][i]][action[j]])
#             object_n[k].append(action[j])
#             short_path[k].append([obs_n[k][i], action[j]])
#
#         max.sort()
#         if max != []:
#             r += max[len(action) - 1]
#         object_n[k].sort()
#         for l in range(len(c)):
#             x_start = lst[obs_n[k][i]][0]
#             y_start = lst[obs_n[k][i]][1]
#             x_end = lst[c[l]][0]
#             y_end = lst[c[l]][1]
#
#
#             plt.annotate('',
#                          xy=(x_end, y_end),  # 箭头终点
#                          xytext=(x_start, y_start),  # 箭头起点
#                          arrowprops=dict(
#                              facecolor='orange',  # 箭头填充色
#                              edgecolor='orange',  # 箭头边缘色
#                              arrowstyle='-|>',  # 箭头样式
#                              lw=2.6,  # 线宽加粗 (原为1)
#                              shrinkA=0,  # 起点不收缩
#                              shrinkB=0,  # 终点不收缩
#                              mutation_scale=26,  # 箭头大小放大 (原为20)
#                              alpha=0.9,  # 添加透明度使更柔和
#                              linestyle='-',  # 实线
#                              connectionstyle='arc3,rad=0.3'  # 添加轻微弧度
#                          ),
#                          zorder=5)  # 确保箭头在最上层
#
#     obs_n[k] = new_obs_n
#     new_obs_n = []
#
# plt.savefig(
#     "D:/python project/TCCN/githup/fig_routing/tree_MST_{}.pdf".format(k),
#     bbox_inches='tight',  # 关键参数：自动裁剪空白
#     pad_inches=0.05,  # 设置最小边距（单位：英寸）
#     dpi=300)  # 保持高分辨率

# #

