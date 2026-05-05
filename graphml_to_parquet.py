import pandas as pd
import xml.etree.ElementTree as ET
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path 

DATA_DIR = Path("data")
GRAPHML_FILE = DATA_DIR / "citation_network_with_topics_new.graphml"
PARQUET_FILE = GRAPHML_FILE.with_suffix(".parquet")

def split_keywords(keywords_value: str):
    """Split a keywords string into individual keywords handling delimiters properly."""
    if keywords_value is None or pd.isna(keywords_value):
        return []
    s = str(keywords_value).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split('|') if p.strip()]
    return parts


def convert_graphml_to_parquet(graphml_file=GRAPHML_FILE, save_path=PARQUET_FILE, save_to_disk=True):
    print(f"Parsing {graphml_file}...")

    tree = ET.parse(graphml_file)
    root = tree.getroot()
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

    keys = {}
    for key in root.findall('.//graphml:key', ns):
        attr_name = key.attrib.get('attr.name')
        attr_id = key.attrib.get('id')
        attr_type = key.attrib.get('attr.type')
        if attr_name and attr_id:
            keys[attr_id] = (attr_name, attr_type)

    print(f"Found attributes: {[v[0] for v in keys.values()]}")

    nodes_data = []
    for node in root.findall('.//graphml:node', ns):
        node_attr = {'node_id': node.attrib.get('id')}
        for data in node.findall('graphml:data', ns):
            key_id = data.attrib.get('key')
            if key_id in keys:
                attr_name, attr_type = keys[key_id]
                val = data.text
                if attr_type == 'double':
                    try:
                        val = float(val) if val else None
                    except ValueError:
                        pass
                node_attr[attr_name] = val
        nodes_data.append(node_attr)

    df = pd.DataFrame(nodes_data)
    print(f"Extracted {len(df)} nodes.")

    if 'keywords' in df.columns:
        print("Converting 'keywords' column to lists...")
        df['keywords'] = df['keywords'].apply(split_keywords)
    if save_to_disk:
        print(f"Saving to {save_path}...")
        df.to_parquet(save_path, index=False)
    print("Done!")
    return df


if __name__ == '__main__':
    convert_graphml_to_parquet()
