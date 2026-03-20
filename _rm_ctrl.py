with open('generate_dummy_data.py', 'rb') as f:
    raw = f.read()
needle = b'"control_flag":   1 - contact_flag,\r\n'
if needle in raw:
    fixed = raw.replace(b'            ' + needle, b'')
    with open('generate_dummy_data.py', 'wb') as f:
        f.write(fixed)
    print('Removed control_flag from generator')
else:
    # try without leading spaces
    idx = raw.find(b'control_flag')
    print(f'Pattern not found as expected. context: {raw[max(0,idx-20):idx+60]}')
