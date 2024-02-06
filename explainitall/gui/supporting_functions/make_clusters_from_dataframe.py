def make_clusters_from_dataframe(df):
    clusters = []
    for row in df.itertuples():
        name = getattr(row, 'name')
        centroid = getattr(row, 'centroid')
        top_k = getattr(row, 'top_k')
        if name is None or name == "":
            continue
        d = {'name': name, 'centroid': eval(centroid), 'top_k': top_k}
        clusters.append(d)
    return clusters
