# -*- coding: utf-8 -*-
#
# @File:   urban_metrics.py
# @Author: Haozhe Xie
# @Date:   2023-08-14 20:08:05
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-08-15 16:10:46
# @Email:  root@haozhexie.com

import argparse
import dijkstar
import json
import logging
import math
import numpy as np
import os
import random

from tqdm import tqdm


def get_connectivity_index(graph):
    n_total_nodes = 0
    n_total_valence = 0
    for nid, values in graph["node"].items():
        ref_ways = values["ref_way"]
        # Ignore nodes whose valence is 2
        if len(ref_ways) == 1 and not _is_ending_nodes(nid, ref_ways[0], graph):
            continue

        n_total_nodes += 1
        # Calculate the valence for the node
        n_valence = 0
        for rw in ref_ways:
            n_valence += 1 if _is_ending_nodes(nid, rw, graph) else 2

        n_total_valence += n_valence if n_valence < 4 else 4

    return n_total_valence / n_total_nodes


def _is_ending_nodes(node_id, way_id, graph):
    node_id = int(node_id) if type(node_id) != int else node_id
    way_id = str(way_id) if type(way_id) != str else way_id

    way_nodes = graph["way"][way_id]["ref_nd"]
    return way_nodes[0] == node_id or way_nodes[-1] == node_id


def get_convenience(graph, n_samples=1000):
    nodes = [int(n) for n in graph["node"].keys()]
    d_graph = _get_d_graph(graph)

    costs = []
    dists = []
    while len(costs) < n_samples:
        n0 = random.choice(nodes)
        n1 = random.choice(nodes)
        if n0 == n1:
            continue
        try:
            path = dijkstar.find_path(d_graph, n0, n1)
        except Exception as ex:
            # logging.exception(ex)
            continue

        costs.append(path.total_cost)
        dists.append(_get_dist(n0, n1, graph))
        # print(costs[-1], dists[-1], n0, n1)

    # assert len(costs) == n_samples, len(costs)
    return np.mean(np.array(dists) / np.array(costs))


def _get_d_graph(graph):
    d_graph = dijkstar.Graph()
    for values in graph["way"].values():
        nodes = values["ref_nd"]
        for i in range(len(nodes) - 1):
            _dist = _get_dist(nodes[i], nodes[i + 1], graph)
            d_graph.add_edge(nodes[i], nodes[i + 1], _dist)
            d_graph.add_edge(nodes[i + 1], nodes[i], _dist)

    return d_graph


def _get_dist(idx0, idx1, graph):
    idx0 = str(idx0) if type(idx0) != str else idx0
    idx1 = str(idx1) if type(idx1) != str else idx1
    pos0 = graph["node"][idx0]["pos"]
    pos1 = graph["node"][idx1]["pos"]
    return math.sqrt((pos0[0] - pos1[0]) ** 2 + (pos0[1] - pos1[1]) ** 2)


def main(input_dir):
    connect_index = []
    convenience = []
    for gf in tqdm([f for f in os.listdir(input_dir) if f.endswith(".json")]):
        with open(os.path.join(input_dir, gf)) as fp:
            graph = json.load(fp)
            connect_index.append(get_connectivity_index(graph))
            convenience.append(get_convenience(graph))

    logging.info("Connectivity Idx: %.4f" % (sum(connect_index) / len(connect_index)))
    logging.info("Convenience: %.4f" % (sum(convenience) / len(convenience)))


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        default=os.getcwd(),
        type=str,
    )
    args = parser.parse_args()

    main(args.input_dir)
