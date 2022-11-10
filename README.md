## generate 5D array
uncomment 
```
I = readimage("data/chessboard_lightfield.png")
L = get_5D(I)
```

## generate manufactured image with focus depth d

```
sub(L,d)
```

## compute all-in-focus image and its depth map

```
I_all, map = all_in(stack, gaussian radius, sigma_1, sigma_2)
```

## generate confocal depth map
```
map = confocal(L)
```

## produce the shift from image to template
```
shift = match(T,I)
```

## refocus using the image stack and shifts
```
output = refocus(L, shifts)
```
