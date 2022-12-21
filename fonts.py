import fontforge, os
F = fontforge.open("C:\All\Stuff\Projects\Images\consola.ttf")
for name in F:
    if F[name].isWorthOutputting() and F[name].unicode >= 0x20 and F[name].unicode <= 0x7a:
        filename = str(F[name].unicode) + ".png"
        # print name
        F[name].export( "C:/All/Stuff/Projects/Images/letters/" + filename, 119)
        # F[name].export(filename, 600)     # set height to 600 pixels