

# %%
hist1, bin_edges1 = np.histogram(pd.DataFrame(mx1).stack(), bins= 50, range= [-0.01, 1.01], density=False)
xbins1 = list(map(lambda i: (bin_edges1[i] + bin_edges1[i-1])/2, range(1,len(bin_edges1))))
hist2, bin_edges2 = np.histogram(pd.DataFrame(mx2).stack(), bins= 50,range= [-0.01, 1.01], density=False)
xbins2 = list(map(lambda i: (bin_edges2[i] + bin_edges2[i-1])/2, range(1,len(bin_edges2))))

xticks1 = list(map(lambda i: "".join([str(round(bin_edges1[i-1],4)), '-', str(round(bin_edges1[i]-.0001, 4))]), range(1,len(bin_edges1))))
xticks2 = list(map(lambda i: "".join([str(round(bin_edges2[i-1],4)), '-', str(round(bin_edges2[i]-.0001, 4))]), range(1,len(bin_edges2))))
t_hist1 = dict(
    type='bar',
    name= 'RDkitFP',
    opacity = 0.8,
    x = xbins1,
    y = hist1,
    marker = dict(color = 'orangered',
    line=dict(width=1,color='orangered')
    )
)

t_hist2 = dict(
    type='bar',
    name= 'MorganFP',
    opacity = 0.8,
    x = xbins2,
    y = hist2,
    marker = dict(color = 'green',
    line=dict(width=1,color='green')
    )
)

layout=dict(
    title='Compound Similarity Distribution',
    xaxis=dict(
        title='Similarity index'
    ),
    yaxis=dict(
        title='Frequency'
    ),
    title_x=0.5,
    barmode='overlay',
    bargap=0,
    annotations=list([
        dict(
            x=1.25,
            y=1.07,
            xref='paper',
            yref='paper',
            text='Fingerprint Algorithm',
            showarrow=False,
        )
    ])
)

figb = go.Figure()
figb.add_traces([t_hist2, t_hist1])

# Overlay both histograms
figb.update_layout(layout)
# Reduce opacity to see both histograms
figb.update_traces(opacity=0.75)
figb.show()
pio.write_html(figb, "figures/Similarity_bar.html")

plot_url_1 = py.plot(fig, filename='Similarity_Hist', auto_open=True,)
print ('Plot Online in {}'.format(plot_url_1))