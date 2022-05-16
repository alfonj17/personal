input_directory="$PWD"
output_directory="${input_directory}/csv"

txt_files=`ls *.txt`
csv_files=`ls *.csv`

 #!Processing csv files
for file in $txt_files
do

   IFS=$'.' read -r file_name cmd <<<"$file"
   input_path="${input_directory}/${file_name}.txt"
   output_path="${output_directory}/${file_name}.csv"

   if test -f "$output_path"; then
      echo "updating $input_path ..."
   else
      touch ${output_path}
      echo "updating $input_path ..."
   fi

#!Here is where the delimeters are defined (first one tab ("/t") and the second one semicolon (";")
   cat  $input_path | tr -s "\t" ";" > $output_path

done

#!Processing csv files
for file in $csv_files
do

   IFS=$'.' read -r file_name cmd <<<"$file"
   input_path="${input_directory}/${file_name}.csv"
   output_path="${output_directory}/${file_name}.csv"

   if test -f "$output_path"; then
      echo "updating $input_path ..."
   else
      touch ${input_path}
      echo "updating $output_path ..."
   fi

#!Here is where the delimeters are defined (first one tab ("/t") and the second one semicolon (";")
   cat  $input_path | tr -s "\t" ";" > $output_path


done
