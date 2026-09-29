# Demos for algorithmic ideas 

Supplemental code examples for my correspondence.
Lack of comments is intentional and so is any obscurity in their purpose.

--------------------------------------

## bitmask4bpm.py

--------------------------------------
## extract_gridROI.py
```bash
python extract_gridROI.py \
    image.npz \
    --nrois 400 \
    --height 100 \
    --width 200 \
    --show=True
```
### output

Input:  116.68 MB \
Output: 0.49 MB \
Size reduction: 99.58% (237.21x smaller) \
ROIs requested/actual: 35/37 
<img width="1116" height="809" alt="Screenshot 2026-09-29 at 3 10 19 PM" src="https://github.com/user-attachments/assets/b2b548c1-0f81-4fb4-b00c-fddbd51690da" />
