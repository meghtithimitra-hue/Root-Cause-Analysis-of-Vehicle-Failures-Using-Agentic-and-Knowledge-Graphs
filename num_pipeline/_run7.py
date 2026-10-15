import sys, os, types
sys.path.insert(0, 'scripts/pipeline')
os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', '')
pkg = types.ModuleType('pipeline')
pkg.__path__ = [os.path.join(os.getcwd(), 'scripts', 'pipeline')]
sys.modules['pipeline'] = pkg
sv_pkg = types.ModuleType('sensor_validation')
sv_pkg.__path__ = [os.path.join(os.getcwd(), 'scripts', 'sensor_validation')]
sys.modules['sensor_validation'] = sv_pkg
from pipeline.hybrid_retrieval import hybrid_retrieve

queries = [
    ('poor engine performance', 'Poor engine performance'),
    ('low engine power', 'Loss of engine power'),
    ('engine consumes too much fuel', 'Poor fuel economy'),
    ('coolant leak', 'Coolant leak'),
    ('engine overheating', 'Engine overheating'),
    ('rough idle', 'Rough idle'),
    ('engine hesitation', 'Engine hesitates on acceleration'),
]
for q, expect in queries:
    r = hybrid_retrieve(q, top_k=10)
    top = r['candidates']
    match_pos = -1
    for i, c in enumerate(top):
        if expect.lower() in c['label'].lower():
            match_pos = i + 1
            break
    print('Q:', q)
    print('  Expected:', expect)
    print('  #1:', top[0]['label'], '(%.3f)' % top[0]['score'], top[0]['source'])
    print('  #2:', top[1]['label'], '(%.3f)' % top[1]['score'], top[1]['source'])
    print('  #3:', top[2]['label'], '(%.3f)' % top[2]['score'], top[2]['source'])
    print('  Match position:', match_pos if match_pos > 0 else 'NOT FOUND')
    print()
