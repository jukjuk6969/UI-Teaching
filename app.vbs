' Peluncur tanpa jendela sama sekali (tidak ada kedipan konsol).
' Klik dua kali file ini, atau buat shortcut-nya ke Desktop.
' Letakkan di folder tool-train (satu folder dengan app.py).
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")

here = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = fso.GetAbsolutePathName(here & "\..\..\venv\Scripts\pythonw.exe")

If Not fso.FileExists(pyw) Then
    MsgBox "Python venv tidak ditemukan di:" & vbCrLf & pyw & vbCrLf & vbCrLf & _
           "Periksa lokasi venv atau ubah baris ""pyw ="" di file ini.", vbCritical, "Labeler"
    WScript.Quit 1
End If

sh.CurrentDirectory = here
sh.Run """" & pyw & """ app.py", 1, False