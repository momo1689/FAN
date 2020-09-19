def load_state_dict_cpu(net, state_dict):
    state_dict1 = net.state_dict()
    for name, value in state_dict1.items():
        assert 'module.'+name in state_dict
        state_dict1[name] = state_dict['module.'+name]
    net.load_state_dict(state_dict1)
