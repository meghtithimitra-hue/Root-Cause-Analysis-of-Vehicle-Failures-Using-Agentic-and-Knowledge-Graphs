import json
from pathlib import Path
from collections import defaultdict

with open('graphify-out/graph.json') as f:
    g = json.load(f)

nodes = g.get('nodes', [])
edges = g.get('links', g.get('edges', []))

# Color map by node type
colors = {
    'cat_':    '#FF6B6B',   # red — Category
    'subcat_': '#4ECDC4',   # teal — Subcategory
    'sym_':    '#45B7D1',   # blue — Symptom
    'step_':   '#96CEB4',   # green — DiagnosisStep
}

# Community colors for category nodes
community_colors = [
    '#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6',
    '#1ABC9C','#E67E22','#34495E','#E91E63','#00BCD4',
    '#8BC34A','#FF5722','#607D8B','#795548','#9C27B0',
    '#03A9F4','#CDDC39'
]

community_palette = [
    '#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6',
    '#1ABC9C','#E67E22','#E91E63','#00BCD4','#8BC34A',
    '#FF5722','#607D8B','#795548','#9C27B0','#03A9F4',
    '#CDDC39','#FF9800'
]

def get_color(node):
    nid = node.get('id', '')
    # Categories always red, large
    if nid.startswith('cat_'):
        return '#FF0040'
    # Everything else colored by community
    cid = node.get('community', 0)
    return community_palette[int(cid) % len(community_palette)]

def get_size(node):
    nid = node.get('id', '')
    if nid.startswith('cat_'):    return 30
    if nid.startswith('subcat_'): return 20
    if nid.startswith('sym_'):    return 10
    if nid.startswith('step_'):   return 8
    return 10

# Build node index
node_map = {n['id']: i for i, n in enumerate(nodes)}

# Build JS data
nodes_js = []
for n in nodes:
    nid = n.get('id', '')
    label = n.get('label', nid)
    display = label if len(label) < 30 else label[:27] + '...'
    nodes_js.append({
        'id': nid,
        'label': display,
        'full_label': label,
        'color': get_color(n),
        'size': get_size(n),
        'community': n.get('community', -1),
        'source_file': n.get('source_file', '')
    })

edges_js = []
edge_colors = {
    'HAS_SUBCATEGORY':   '#FF6B6B',
    'HAS_SYMPTOM':       '#45B7D1',
    'HAS_DIAGNOSIS_STEP':'#96CEB4',
    'SIMILAR_SYMPTOM_TO':'#F39C12',
    'SIMILAR_STEP_TO':   '#9B59B6',
}
for e in edges:
    src = e.get('source', '')
    tgt = e.get('target', '')
    rel = e.get('relation', '')
    if src in node_map and tgt in node_map:
        edges_js.append({
            'source': src,
            'target': tgt,
            'relation': rel,
            'color': edge_colors.get(rel, '#BDC3C7'),
            'width': 3 if 'SIMILAR' in rel else 1
        })

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vehicle Fault Knowledge Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1a1a2e; font-family: Arial, sans-serif; overflow: hidden; }}
#canvas {{ width: 100vw; height: 100vh; }}
#tooltip {{
    position: absolute; background: rgba(0,0,0,0.85);
    color: white; padding: 10px 14px; border-radius: 8px;
    font-size: 13px; pointer-events: none; display: none;
    border: 1px solid #444; max-width: 300px; line-height: 1.6;
}}
#legend {{
    position: absolute; top: 20px; left: 20px;
    background: rgba(0,0,0,0.7); color: white;
    padding: 14px; border-radius: 8px; font-size: 13px;
}}
#legend h3 {{ margin-bottom: 8px; color: #eee; }}
.legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }}
#stats {{
    position: absolute; top: 20px; right: 20px;
    background: rgba(0,0,0,0.7); color: white;
    padding: 14px; border-radius: 8px; font-size: 13px;
}}
#controls {{
    position: absolute; bottom: 20px; left: 50%;
    transform: translateX(-50%);
    background: rgba(0,0,0,0.7); color: white;
    padding: 10px 20px; border-radius: 8px; font-size: 12px;
    text-align: center;
}}
</style>
</head>
<body>
<svg id="canvas"></svg>
<div id="tooltip"></div>
<div id="legend">
    <h3>Node Types</h3>
    <div class="legend-item"><div class="legend-dot" style="background:#FF6B6B"></div>Category (13)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4ECDC4"></div>Subcategory (98)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#45B7D1"></div>Symptom (143)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#96CEB4"></div>Diagnosis Step (190)</div>
    <h3 style="margin-top:10px">Edge Types</h3>
    <div class="legend-item"><div class="legend-dot" style="background:#FF6B6B;border-radius:0;height:3px;width:20px;margin-right:4px"></div>Has Subcategory</div>
    <div class="legend-item"><div class="legend-dot" style="background:#45B7D1;border-radius:0;height:3px;width:20px;margin-right:4px"></div>Has Symptom</div>
    <div class="legend-item"><div class="legend-dot" style="background:#96CEB4;border-radius:0;height:3px;width:20px;margin-right:4px"></div>Has Diagnosis Step</div>
    <div class="legend-item"><div class="legend-dot" style="background:#F39C12;border-radius:0;height:3px;width:20px;margin-right:4px"></div>Similar Symptom (cross-system)</div>
</div>
<div id="stats">
    <b>Graph Stats</b><br>
    Nodes: {len(nodes)}<br>
    Edges: {len(edges)}<br>
    Communities: 17<br>
    Multi-category: 13/17 (76.5%)<br>
    Cross-category edges: 203
</div>
<div id="controls">
    🖱 Drag nodes • Scroll to zoom • Click node for details • Double-click to pin/unpin
</div>

<script>
const nodesData = {json.dumps(nodes_js)};
const edgesData = {json.dumps(edges_js)};

const width = window.innerWidth;
const height = window.innerHeight;
const tooltip = document.getElementById('tooltip');

const svg = d3.select('#canvas')
    .attr('width', width)
    .attr('height', height);

const g = svg.append('g');

svg.call(d3.zoom()
    .scaleExtent([0.1, 8])
    .on('zoom', e => g.attr('transform', e.transform)));

const sim = d3.forceSimulation(nodesData)
    .force('link', d3.forceLink(edgesData)
        .id(d => d.id)
        .distance(d => d.relation && d.relation.includes('SIMILAR') ? 200 : 60)
        .strength(d => d.relation && d.relation.includes('SIMILAR') ? 0.1 : 0.8))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('x', d3.forceX(width/2).strength(0.05))
    .force('y', d3.forceY(height/2).strength(0.05))
    .force('collision', d3.forceCollide().radius(d => d.size + 4));

const link = g.append('g')
    .selectAll('line')
    .data(edgesData)
    .join('line')
    .attr('stroke', d => d.color)
    .attr('stroke-width', d => d.width || 1)
    .attr('stroke-opacity', 0.6);

const node = g.append('g')
    .selectAll('circle')
    .data(nodesData)
    .join('circle')
    .attr('r', d => d.size)
    .attr('fill', d => d.color)
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.5)
    .attr('opacity', 0.9)
    .call(d3.drag()
        .on('start', (e, d) => {{
            if (!e.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
        }})
        .on('drag', (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
        .on('end', (e, d) => {{
            if (!e.active) sim.alphaTarget(0);
        }}))
    .on('mouseover', (e, d) => {{
        const sf = d.source_file.split('/').slice(-2).join('/');
        tooltip.style.display = 'block';
        tooltip.innerHTML = `<b>${{d.full_label}}</b><br>
            Type: ${{d.id.split('_')[0]}}<br>
            Community: ${{d.community}}<br>
            Source: ${{sf}}`;
    }})
    .on('mousemove', e => {{
        tooltip.style.left = (e.pageX + 12) + 'px';
        tooltip.style.top  = (e.pageY - 10) + 'px';
    }})
    .on('mouseout', () => {{ tooltip.style.display = 'none'; }})
    .on('dblclick', (e, d) => {{
        if (d.fx !== null) {{ d.fx = null; d.fy = null; }}
        else {{ d.fx = d.x; d.fy = d.y; }}
    }});

const label = g.append('g')
    .selectAll('text')
    .data(nodesData.filter(d => d.id.startsWith('cat_') || d.id.startsWith('subcat_')))
    .join('text')
    .text(d => d.label)
    .attr('font-size', d => d.id.startsWith('cat_') ? 13 : 9)
    .attr('fill', '#fff')
    .attr('text-anchor', 'middle')
    .attr('dy', d => -(d.size + 4))
    .attr('pointer-events', 'none');

sim.on('tick', () => {{
    link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
    node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
    label
        .attr('x', d => d.x)
        .attr('y', d => d.y);
}});
</script>
</body>
</html>"""

out_path = Path('graphify-out/graph_viz.html')
out_path.write_text(html, encoding='utf-8')
print(f"Saved: {out_path}")
print("Opening in browser...")
import webbrowser, os
webbrowser.open(f"file:///{os.path.abspath(out_path)}")