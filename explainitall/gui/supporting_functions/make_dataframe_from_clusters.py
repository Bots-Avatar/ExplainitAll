def make_dataframe_from_clusters(clusters):
    clusters = copy.deepcopy(clusters)
    for c in clusters:
        #c['centroid'] = str(c['centroid'])
        c['centroid'] = "['" + "', '".join(c['centroid']) + "']"
    res = pd.DataFrame(clusters)
    return res
