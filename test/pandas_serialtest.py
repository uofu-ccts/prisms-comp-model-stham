import pandas as pd;
import types;


data = []

for i in range(0,3):
x = types.SimpleNamespace();
x.a = 1;
x.b = 'a'
x.c = 1.04
#   x = []
#   x += [1]
#   x += ['a']
#   x += [1.04]
  data += [x]


print(data)

pddata = pd.DataFrame(data)

print(pddata)


# out = open('temp.d','wb')
# pddata.to_csv(out);
# out.close()
# inp = open('temp.d','rb')
# a = pd.from_csv(inp);
# inp.close()
# print(a)

# # #out = open('temp.d','wb')
# pddata.to_hdf('temp.hdf','arg');
# # #out.close()
# # #inp = open('temp.d','rb')
# a = pd.from_hdf('temp.hdf', 'arg');
# # #inp.close()
# print(a)

#out = open('temp.d','wb')
pddata.to_pickle('temp.d');
#out.close()
#inp = open('temp.d','rb')
a = pd.read_pickle('temp.d');
#inp.close()
print(a)
print(a.iloc[0])
print(type(a.iloc[0][0]))
print(a.iloc[0][0].a)
print(type(a.iloc[0][0].a))


