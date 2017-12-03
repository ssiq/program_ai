import os
import logging
import objgraph
import gc
_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    gc.collect()
    return _VmB('VmSize:') - since


def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since


def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since


def record_memory(loggername:str):
    mem = memory()
    logger = logging.getLogger(loggername)
    logger.debug('memory size : {}'.format(mem))

def show_growth(recordloggername:str, growthloggername:str, peak_stats={}, shortnames=True):

    def get_count(one):
        return one[1]['delta']

    growthlogger = logging.getLogger(growthloggername)
    recordlogger = logging.getLogger(recordloggername)

    gc.collect()
    stats = objgraph.typestats(shortnames=shortnames)
    deltas = {}
    max_name = None
    max_delta = 0
    for name, count in objgraph.iteritems(stats):
        old_count = peak_stats.get(name, 0)
        if count > old_count:
            deltas[name] = {}
            delta = count - old_count
            deltas[name]['name'] = name
            deltas[name]['delta'] = delta
            deltas[name]['count'] = stats[name]
            if delta > max_delta:
                max_name = name
                max_delta = delta
            peak_stats[name] = count
    deltas = sorted(deltas.items(), key=get_count, reverse=True)

    if len(growthlogger.handlers) <= 0:
        print('growthlogger handler length error')
        return
    fh = growthlogger.handlers[0]
    filename = fh.baseFilename

    import json
    growthlogger.debug(json.dumps(deltas))
    recordlogger.debug('record object growth count in {}. max growth type {} and growth {}'.format(os.path.abspath(filename), max_name, max_delta))


def show_diff_length_fn(recordloggername:str, lengthloggername:str):

    list_length = {}
    dict_length = {}
    lengthlogger = logging.getLogger(lengthloggername)
    recordlogger = logging.getLogger(recordloggername)

    if len(lengthlogger.handlers) <= 0:
        print('lengthlogger handler length error')
        return
    fh = lengthlogger.handlers[0]
    filename = fh.baseFilename

    def get_delta(one):
        return one[1]['delta']

    def show_diff_len(objs, global_record:dict, min_record_delta_length=100, i=0):
        type_name = type(objs[0]).__name__
        recordlogger.debug('record {} delta count in {}.'.format(type_name, filename))

        cur_list_length = {}
        cur_add_length = {}
        for obj in objs:
            address = hex(id(obj))
            tmp = {}
            tmp['name'] = address
            tmp['count'] = len(obj)
            if address in global_record.keys():
                old = global_record[address]['count']
                delta = len(obj) - old
                tmp['delta'] = delta
                if delta > 0 and i != 0:
                    cur_add_length[address] = tmp
                if delta > min_record_delta_length and i != 0:
                    dot_path, obj_len = record_obj_ref(obj)
                    if obj_len > 1:
                        recordlogger.debug(
                            'record {} object {} in dot path {} . path items length: {}. object length : {}. object delta length : {}'.format(type_name,
                                                                                                                       address, dot_path, obj_len, len(obj), delta))
            else:
                delta = len(obj)
                tmp['delta'] = delta
                if delta > 0 and i != 0:
                    cur_add_length[address] = tmp
            cur_list_length[address] = tmp
        cur_add_length = sorted(cur_add_length.items(), key=get_delta, reverse=True)
        a = {type_name: cur_add_length}

        import json
        from code_data.constants import OBJECTLENGTH_PATH
        import time
        if not os.path.exists(OBJECTLENGTH_PATH):
            os.makedirs(OBJECTLENGTH_PATH)
        str_time = time.strftime('_%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
        name = '{}_{}_{}.log'.format(type_name, 'objectlength', str_time)
        ol_filepath = os.path.join(OBJECTLENGTH_PATH, name)
        ol_file = open(ol_filepath, 'w')
        ol_file.write(json.dumps(a))
        ol_file.flush()
        ol_file.close()

        global_record.clear()
        for name, i in cur_list_length.items():
            global_record[name] = i

    def show_diff_length(min_record_delta_length=100, i=0):
        gc.collect()
        objs = objgraph.by_type(list.__name__)
        show_diff_len(objs, list_length, min_record_delta_length, i)

        gc.collect()
        objs = objgraph.by_type(dict.__name__)
        show_diff_len(objs, dict_length, i)
        gc.collect()

    return show_diff_length



def record_obj_ref(obj):
    import time
    from code_data.constants import DOT_PATH
    address = hex(id(obj))
    str_time = time.strftime('_%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    name = '{}_{}.dot'.format(address, str_time)
    if not os.path.exists(os.path.abspath(DOT_PATH)):
        os.makedirs(DOT_PATH)
    file_path = os.path.join(DOT_PATH, name)
    objs = objgraph.find_backref_chain(obj, objgraph.is_proper_module, max_depth=20)
    if len(objs) > 1:
        objgraph.show_chain(objs, filename=file_path)
    return file_path, len(objs)


