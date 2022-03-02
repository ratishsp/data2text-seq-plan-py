import argparse
import os
import os.path
from shutil import copyfile

def process(step_file_name, summary_file_name, pp_inp_file_name, pp_out_file_name):
    step_file = open(step_file_name, mode='r')
    step_file = [x.strip() for x in step_file]
    output_file = open(summary_file_name + ".tmp", mode='w')
    output_lines = []
    if not os.path.isfile(summary_file_name):  # first step
        for step_file_line in step_file:
            output_lines.append("<segment> " + step_file_line)
    else:
        summary_file = open(summary_file_name, mode='r')
        summary_file = [x.strip() for x in summary_file]
        for summary_file_line, step_file_line in zip(summary_file, step_file):
            output_lines.append(" <segment> ".join([summary_file_line, step_file_line]))
    output_file.write("\n".join(output_lines))
    output_file.write("\n")
    output_file.close()
    os.rename(summary_file_name+".tmp", summary_file_name)

    output_lines = []
    pp_inp_tmp_file = open(pp_inp_file_name + ".tmp", mode='w')
    pp_out_file = open(pp_out_file_name, mode='r')
    pp_out_file = [x.strip() for x in pp_out_file]
    if not os.path.isfile(pp_inp_file_name):  # first step
        copyfile(pp_out_file_name, pp_inp_file_name)
    else:
        pp_inp_file = open(pp_inp_file_name, mode='r')
        pp_inp_file = [x.strip() for x in pp_inp_file]
        for pp_inp_file_line, pp_out_file_line in zip(pp_inp_file, pp_out_file):
            output_lines.append(" ".join([pp_inp_file_line, pp_out_file_line]))
        pp_inp_tmp_file.write("\n".join(output_lines))
        pp_inp_tmp_file.write("\n")
        pp_inp_tmp_file.close()
        os.rename(pp_inp_file_name + ".tmp", pp_inp_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocessing step output to summary')
    parser.add_argument('-step_file', type=str,
                        help='path of step file', default=None)
    parser.add_argument('-summary_file', type=str,
                        help='path of summary file', default=None)
    parser.add_argument('-pp_inp_file', type=str,
                        help='path of pp input file', default=None)
    parser.add_argument('-pp_out_file', type=str,
                        help='path of pp output file', default=None)
    args = parser.parse_args()

    process(args.step_file, args.summary_file, args.pp_inp_file, args.pp_out_file)
