import urllib.request, json, time, sys

try:
    start = time.time()
    req = urllib.request.Request(
        'http://localhost:8000/api/ask', 
        data=json.dumps({'question': 'Việt Nam là gì?', 'reader': 'llm', 'retriever': 'bm25', 'top_k': 3}).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    res = urllib.request.urlopen(req, timeout=300)
    print("Success:", res.read().decode('utf-8')[:200])
    print(f'Time: {time.time()-start}')
except Exception as e:
    if hasattr(e, 'read'):
        print(f"Error {e}: {e.read().decode('utf-8')}")
    else:
        print(f"Error: {e}")
