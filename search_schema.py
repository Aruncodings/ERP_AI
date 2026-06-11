import json

def main():
    filepath = r"c:\Users\sadmin\Downloads\Arun\train\temp_fields.json"
    encodings = ["utf-16", "utf-16-le", "utf-16-be", "utf-8"]
    
    data = None
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                content = f.read()
                data = json.loads(content)
                print(f"Successfully loaded using {enc}")
                break
        except Exception as e:
            continue
            
    if not data:
        print("Could not load temp_fields.json with any encoding.")
        return
        
    print("Searching for 'escalationLevel' in schema:")
    # data can be a list or dict
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []
        
    found_count = 0
    for idx, item in enumerate(items):
        item_str = json.dumps(item)
        if "escalationLevel" in item_str:
            found_count += 1
            print(f"\n--- Match {found_count} (index {idx}) ---")
            # print field metadata
            if isinstance(item, dict):
                # find where escalationLevel is
                for key, val in item.items():
                    if key == "fields" and isinstance(val, list):
                        for field in val:
                            if field.get("field") == "escalationLevel":
                                print(json.dumps(field, indent=2))
                    elif key == "escalationLevel":
                        print(f"Key: escalationLevel = {val}")
                    elif isinstance(val, (dict, list)) and "escalationLevel" in json.dumps(val):
                        # print partial
                        print(f"Key {key}: contains escalationLevel")
                        if isinstance(val, list):
                            for f in val:
                                if isinstance(f, dict) and f.get("field") == "escalationLevel":
                                    print(json.dumps(f, indent=2))

if __name__ == "__main__":
    main()
