import sys, os, types
sys.path.insert(0, 'scripts/pipeline')
os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', '')
pkg = types.ModuleType('pipeline')
pkg.__path__ = [os.path.join(os.getcwd(), 'scripts', 'pipeline')]
sys.modules['pipeline'] = pkg
sv_pkg = types.ModuleType('sensor_validation')
sv_pkg.__path__ = [os.path.join(os.getcwd(), 'scripts', 'sensor_validation')]
sys.modules['sensor_validation'] = sv_pkg
from pipeline.query_preprocessor import preprocess_query

for q in ['engine consumes too much fuel', 'rough idle', 'engine hesitation']:
    p = preprocess_query(q)
    print('='*70)
    print('Query:', q)
    print('Processed:', p['processed'])
    print('Entities:', [(e['label'], round(e['confidence'],3), e.get('community_id')) for e in p['entities']])
    print('Expanded:', p['expanded_queries'])
    print('Hints:', p['retrieval_hints'])
    print()
