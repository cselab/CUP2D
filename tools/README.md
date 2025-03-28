Remove comments

```
gcc -fpreprocessed -dD -E -P Windmill.h
```

```
unifdef -UZERO_TOTAL_MOM -m `find . -name '*.cpp' -or -name '*.h' -or -name '*.cu' -or -name '*.cuh'`
```

```
clang-format -i `find . -name '*.cpp' -or -name '*.h' -or -name '*.cu' -or -name '*.cuh'`
```