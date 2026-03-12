from med_bot.config import SUPABASE_KEY, SUPABASE_TABLE_NAME, SUPABASE_URL
from supabase import create_client


def load_table():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase.table(SUPABASE_TABLE_NAME)


# Returns the rows in the data filtered by the command.
# Outputs a list of dictionaries whose keys are the table's columns
# and values are the values in the respective cells.
def sql_command_table(table, command):
    response = table.select(command).execute()
    return response.data
