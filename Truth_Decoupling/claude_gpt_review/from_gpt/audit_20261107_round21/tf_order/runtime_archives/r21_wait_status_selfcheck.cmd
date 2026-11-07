python3 - <<'PY'
import os,pathlib,time
pid=os.fork()
if pid==0:os._exit(7)
for _ in range(100):
 content=pathlib.Path(f'/proc/{pid}/stat').read_text();fields=content[content.rfind(')')+2:].split()
 if fields[0]=='Z':
  status=int(fields[49]);actual=os.waitpid(pid,0)[1]
  assert status==actual and os.waitstatus_to_exitcode(status)==7
  print('wait_status decoding actual child exit7 verified');break
 time.sleep(.01)
else:raise RuntimeError('child did not exit')
PY
